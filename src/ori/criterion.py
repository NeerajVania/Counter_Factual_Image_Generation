import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

def iou_loss(pred, y):
    n = y.numel()
    pred = torch.sigmoid(pred)

    a = (pred * y).sum()
    b = ((1 - pred) * (1 - y)).sum()
    iou_true = a / (n - b)

    return 1. - iou_true

def mse_loss(pred, y):
    pred = torch.sigmoid(pred)
    return ((pred - y)**2).mean()

def vae_loss(recon_x, x, mu, logvar, lpips_fn, beta=1e-6):
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss
    return total_loss


def vae_loss_with_lpips(recon_x, x, mu, logvar, lpips_fn, beta, lpips_weight):
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    

    mean = torch.tensor([0.3704248070716858, 0.2282254546880722, 0.13915641605854034]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.23381589353084564, 0.1512117236852646, 0.09653093665838242]).view(1, 3, 1, 1).to(x.device)

    x_denorm = x * std + mean
    recon_x_denorm = recon_x * std + mean
    
    x_norm = torch.clamp(x_denorm, 0, 1) * 2.0 - 1.0
    recon_x_norm = torch.clamp(recon_x_denorm, 0, 1) * 2.0 - 1.0
    
    lpips_loss = torch.mean(lpips_fn(recon_x_norm, x_norm))
    total_loss = recon_loss + beta * kl_loss + lpips_weight * lpips_loss
    return total_loss, {'recon_loss': recon_loss, 'kl_loss': kl_loss, 'lpips_loss': lpips_loss}

def coral_loss(logits, levels, importance_weights=None, reduction='mean'):
    """Computes the CORAL loss described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        Outputs of the CORAL layer.

    levels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors
        (via `coral_pytorch.dataset.levels_from_labelbatch`).

    importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
        Optional weights for the different labels in levels.
        A tensor of ones, i.e.,
        `torch.ones(num_classes-1, dtype=torch.float32)`
        will result in uniform weights that have the same effect as None.

    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).

    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import coral_loss
    >>> levels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> coral_loss(logits, levels)
    tensor(0.6920)
    """

    if not logits.shape == levels.shape:
        raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                         % (logits.shape, levels.shape))

    term1 = (F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))

    if importance_weights is not None:
        term1 *= importance_weights

    val = (-torch.sum(term1, dim=1))

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss


class NT_Xent(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)  # dynamically adjust for the batch size
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        mask = self.mask_correlated_samples(batch_size).to(z.device)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(-1, 1)
        negative_samples = sim[mask].view(N, -1)

        labels = torch.zeros(N, device=positive_samples.device, dtype=torch.long)
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class ODloss(nn.Module):
    def __init__(self, timesteps):
        super().__init__()
        self.T = timesteps

    def forward(self, noise_pred, noise, labels, timestep, ordinal: bool):
        mse_loss = nn.functional.mse_loss(noise_pred, noise)

        if not ordinal:
            return mse_loss

        lam = timestep.float().mean() / self.T

        labels = labels.view(-1)

        li = labels.view(-1, 1, 1)
        lj = labels.view(1, -1, 1)
        lk = labels.view(1, 1, -1)

        valid = (li < lj) & (lj < lk)
        valid_indices = valid.nonzero(as_tuple=False)

        if valid_indices.size(0) == 0:
            return mse_loss

        i, j, k = valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
        
        # L2 norm
        d_ij = torch.linalg.norm(noise_pred[i] - noise_pred[j], dim=1)
        d_jk = torch.linalg.norm(noise_pred[j] - noise_pred[k], dim=1)
        d_ik = torch.linalg.norm(noise_pred[i] - noise_pred[k], dim=1)

        ord_term = (d_ik - (d_ij + d_jk)) ** 2
        ordinal_loss = lam * ord_term.mean()

        total_loss = mse_loss + 0.01 * ordinal_loss

        return total_loss


def get_criterion(cfg):
    criteria = {
        'BCE': F.binary_cross_entropy_with_logits,
        'CCE': F.cross_entropy,
        'IoU': iou_loss,
        #'MSE': mse_loss, #lambda pred, y: F.mse_loss(torch.sigmoid(pred), y),
        'MSE': F.mse_loss,
        'VAE': vae_loss,
        'VAE_LPIPS': vae_loss_with_lpips,
        'coral': coral_loss,
        'NT_Xent': NT_Xent(),
        #'ODloss': ODloss(cfg.noise_scheduler.num_timesteps),
        'velocity_mse': F.mse_loss,
    }


    return criteria[cfg.training.criterion]

