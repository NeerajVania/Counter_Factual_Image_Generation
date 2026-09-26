from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


name_to_optimizer = {
    'Adam': Adam,
    'AdamW': AdamW,
    'SGD': SGD,
}

name_to_scheduler = {
    'CosineAnnealing': CosineAnnealingLR,
}


def get_optimizer(name):
    return name_to_optimizer[name]

def get_scheduler(name):
    return name_to_scheduler[name]
