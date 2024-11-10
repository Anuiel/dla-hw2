import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from src.metrics.base_metric import BaseMetric


class SISignalNoiseRatio(BaseMetric):
    def __init__(self, compute_on_cpu: bool = True, *args, **kwargs):
        """
        Args:
            compute_on_cpu: compute si-snr of cpu or gpu
        """
        super().__init__(*args, **kwargs)
        self.si_snr = ScaleInvariantSignalNoiseRatio(compute_on_cpu=compute_on_cpu)

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        result = []
        for pred, target in zip(preds, targets):
            result.append(self.si_snr(pred, target).detach().item())
        return sum(result) / len(result)
