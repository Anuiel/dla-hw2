import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from src.metrics.base_metric import BaseMetric


class PESQ(BaseMetric):
    def __init__(
        self, device: str, mode: str, sampling_rate: int = 16000, *args, **kwargs
    ):
        """
        Args:
            compute_on_cpu: compute si-snr of cpu or gpu
        """
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(sampling_rate, mode).to(
            device=device
        )

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        return self.pesq(preds, targets).detach().mean().item()
