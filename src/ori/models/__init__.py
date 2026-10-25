from .unet import UNet
from .DiT import DiT
from .autoencoder import VAE
from .resnet import resnet
from .encoder import encoder

def get_model(cfg):
    if cfg.model.name == "DiT":
        learn_sigma = getattr(cfg.model, 'learn_sigma', True)
        # Support both old yaml (equi/learned/structure) and new RF yaml
        # (ordinal_input/learned_ordinal_input/use_structure)
        ordinal_input = getattr(cfg.model, 'ordinal_input',
                                getattr(cfg.model, 'equi', False))
        learned_ordinal_input = getattr(cfg.model, 'learned_ordinal_input',
                                        getattr(cfg.model, 'learned', True))
        use_structure = getattr(cfg.model, 'use_structure',
                                getattr(cfg.model, 'structure', True))
        return DiT(
        input_size=cfg.data.image_resolution // 4,
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.in_channels,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        learn_sigma=learn_sigma,
        ordinal_input=ordinal_input,
        learned_ordinal_input=learned_ordinal_input,
        use_structure=use_structure,
        )
    elif cfg.model.name == 'resnet':
        return resnet(
            num_classes=5,
            pretrained=True,
            classification_type=cfg.training.criterion
        )
    elif cfg.model.name == 'VAE':
        return VAE(
            in_channels=3,
            base_channels=cfg.model.base_channels,
            ch_mult=cfg.model.ch_mult,
            num_res_blocks=cfg.model.num_res_blocks,
            latent_channels=cfg.model.latent_channels
        )
    elif cfg.model.name == 'encoder':
        return encoder(
        )
    


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)
