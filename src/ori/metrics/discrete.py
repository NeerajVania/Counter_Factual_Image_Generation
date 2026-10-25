import torch

from torchmetrics import ConfusionMatrix as TorchMetricsConfusionMatrix
from .abstract import Metric


class ConfusionMatrix(Metric):
    def __init__(self, num_classes):
        assert num_classes >= 2
        super().__init__()

        if num_classes==2:
            task = 'binary'
        else:
            task = 'multiclass'

        self.num_classes = num_classes
        self._confusion_matrix = TorchMetricsConfusionMatrix(num_classes=num_classes, task=task)
        

    def reset(self):
        self._confusion_matrix.reset()

    def update(self, _, groundtruth, prediction):
        self._confusion_matrix.update(preds=prediction, target=groundtruth)

    def get_confusion_matrix(self):
        return self._confusion_matrix.compute()

    def get_accuracy(self):
        confusion_counts = self.get_confusion_matrix()
        return confusion_counts.diag().sum() / confusion_counts.sum()

    def get_balanced_accuracy(self):
        confusion_counts = self.get_confusion_matrix()
        recalls = confusion_counts.diag() / confusion_counts.sum(axis=1)
        return recalls.mean()

    def get_ious(self):
        confusion_counts = self.get_confusion_matrix()
        col_sums = confusion_counts.sum(axis=0)
        row_sums = confusion_counts.sum(axis=1)
        diag = confusion_counts.diag()
        return diag / (row_sums + col_sums - diag)

    def get_miou(self):
        return self.get_ious().mean()

    def get_quadratic_weighted_kappa(self):
        """
        Compute Quadratically Weighted Kappa (QWK) using the confusion matrix.
        """
        confusion_counts = self.get_confusion_matrix().float()
        C = self.num_classes
        total = confusion_counts.sum()

        W = torch.zeros((C, C), device=confusion_counts.device, dtype=confusion_counts.dtype)
        for i in range(C):
            for j in range(C):
                W[i, j] = ((i - j) ** 2) / ((C - 1) ** 2)

        O = confusion_counts / total

        row_marginals = confusion_counts.sum(dim=1) / total
        col_marginals = confusion_counts.sum(dim=0) / total
        E = torch.ger(row_marginals, col_marginals)

        numerator = (W * O).sum()
        denominator = (W * E).sum()

        kappa = 1 - numerator / (denominator + 1e-8)
        return kappa


    def _get_key_to_eval_func(self):
        return {
            "Confusion Matrix": self.get_confusion_matrix,
            "Accuracy": self.get_accuracy,
            "Balanced Accuracy": self.get_balanced_accuracy,
            "IoUs": self.get_ious,
            "Mean IoU": self.get_miou,
            "Kappa": self.get_quadratic_weighted_kappa,
        }


    @classmethod
    def from_cfg(cls, cfg):
        return cls(num_classes=cfg["num_classes"])
