from abc import ABC, abstractmethod
from torch.nn import Module

class Metric(Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def get_out_dict(self, keys=None):
        key_to_eval_func = self._get_key_to_eval_func()
        if keys is None:
            keys = key_to_eval_func.keys()
        return {key: key_to_eval_func[key]() for key in keys}

    @classmethod
    def get_keys(cls):
        return cls.__new__(cls)._get_key_to_eval_func().keys()

    @classmethod
    def from_cfg(cls, cfg, device):
        return cls(device)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, x, groundtruth, prediction):
        pass

    @abstractmethod
    def _get_key_to_eval_func(self):
        pass
