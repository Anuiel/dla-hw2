import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from src.loss import PIT_SISNR
from src.metrics.base_metric import BaseMetric


class SISignalNoiseRatio(BaseMetric):
    def __init__(
        self,
        device: str,
        use_pit: bool = True,
        n_speakers: int | None = None,
        *args,
        **kwargs
    ):
        """
        Args:
            compute_on_cpu: compute si-snr of cpu or gpu
        """
        super().__init__(*args, **kwargs)
        self.use_pit = use_pit
        if not use_pit:
            self.si_snr = ScaleInvariantSignalNoiseRatio().to(device=device)
        else:
            assert (
                use_pit and n_speakers is not None
            ), "Must provide speakers count when using PIT"
            self.si_snr = PIT_SISNR(n_speakers, return_dict=False).to(device=device)

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        # Does averaging on torchmetrics' side
        if self.use_pit:
            return -self.si_snr(preds, targets)
        return self.si_snr(preds, targets).detach().item()
