from torch.nn import ModuleList

from .abstract import Metric
from .continuous import MeanAbsoluteError, MeanSquaredError
from .discrete import ConfusionMatrix


_metrics = [
    MeanAbsoluteError,
    MeanSquaredError,
    ConfusionMatrix,
]

_key_to_metric = {key: metric for metric in _metrics for key in metric.get_keys()}


class Metrics(ModuleList, Metric):
    def __init__(self, metrics, keys):
        super().__init__(metrics)
        self.keys = keys

    def reset(self):
        for metric in self:
            metric.reset()

    def update(self, x, groundtruth, prediction):
        for metric in self:
            metric.update(x, groundtruth, prediction)

    def _get_key_to_eval_func(self, keys=None):
        if keys is None:
            keys = self.keys
        return {
            key: eval_function
            for metric in self
            for key, eval_function in metric._get_key_to_eval_func().items()
            if key in keys
        }


def get_metrics(cfg):
    metrics = {_key_to_metric[key] for key in cfg["metrics"]}
    return Metrics([metric.from_cfg(cfg) for metric in metrics], cfg["metrics"])
