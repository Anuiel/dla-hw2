import torch
from torchmetrics.audio import SignalDistortionRatio

from src.metrics.base_metric import BaseMetric

class SDR(BaseMetric):
    def __init__(self, device: str, *args, **kwargs):
        """
        Args:
            compute_on_cpu: compute si-snr of cpu or gpu
        """
        super().__init__(*args, **kwargs)
        self.sdr = SignalDistortionRatio().to(device=device)

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        # Does averaging on torchmetrics' side
        return self.sdr(preds, targets).detach().item()
