import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from src.metrics.base_metric import BaseMetric


class SISNRi(BaseMetric):
    def __init__(self, device: str, *args, **kwargs):
        """
        Args:
            compute_on_cpu: compute si-snr of cpu or gpu
        """
        super().__init__(*args, **kwargs)
        self.si_snr = ScaleInvariantSignalNoiseRatio().to(device=device)

    def __call__(
        self,
        mix_audio: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ):
        """
        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        # Does averaging on torchmetrics' side
        naive_pred = mix_audio.unsqueeze(1).repeat(1, 2, 1)
        return (
            self.si_snr(preds, targets).detach().item()
            - self.si_snr(naive_pred, targets).detach().item()
        )
