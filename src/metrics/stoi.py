import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric

class STOI(BaseMetric):
    def __init__(self, device: str, sampling_rate: int = 16000, *args, **kwargs):
        """
        Args:
            compute_on_cpu: compute si-snr of cpu or gpu
        """
        super().__init__(*args, **kwargs)
        self.pesq = ShortTimeObjectiveIntelligibility(sampling_rate).to(device=device)

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        return self.pesq(preds, targets).detach().item()
