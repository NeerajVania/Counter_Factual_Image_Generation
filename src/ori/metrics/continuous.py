import torch
from .abstract import Metric


class MeanAbsoluteError(Metric):
    def __init__(self):
        super().__init__()

    def reset(self):
        self._abs_error = 0.0
        self._num_samples = 0

    def update(self, _, groundtruth, prediction):
        pred_flat = prediction.reshape(-1)
        gt_flat = groundtruth.reshape(-1)
        self._abs_error += torch.abs(pred_flat - gt_flat).sum()
        self._num_samples += len(pred_flat)

    def get_mae(self):
        if self._num_samples > 0:
            return self._abs_error / self._num_samples
        else:
            return None

    def _get_key_to_eval_func(self):
        return {"MAE": self.get_mae}


class MeanSquaredError(Metric):
    def __init__(self):
        super().__init__()

    def reset(self):
        self._squared_error = 0.0
        self._num_samples = 0

    def update(self, _, groundtruth, prediction):
        pred_flat = prediction.reshape(-1)
        gt_flat = groundtruth.reshape(-1)
        self._squared_error += torch.sum((pred_flat - gt_flat) ** 2)
        self._num_samples += len(pred_flat)

    def get_mse(self):
        if self._num_samples > 0:
            return self._squared_error / self._num_samples
        else:
            return None

    def get_psnr(self):
        if self._num_samples > 0:
            return -10 * torch.log10(self.get_mse())
        else:
            return None

    def _get_key_to_eval_func(self):
        return {
            "MSE": self.get_mse,
            "PSNR": self.get_psnr,
        }
