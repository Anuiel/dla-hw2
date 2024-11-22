import torch
from torch import nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SISNR(nn.Module):
    """
    Wrapper around SI-SNR
    """

    def __init__(self, return_dict: bool = True):
        super().__init__()
        self.return_dict = return_dict
        self.si_snr = ScaleInvariantSignalNoiseRatio()

    def forward(
        self, pred: torch.Tensor, target_audio: torch.Tensor, **batch
    ) -> dict | float:
        """
        Loss function calculation logic.

        Args:
            pred: Tensor, [batch_size, seq_len]
            target_audio: Tensor, [batch_size, seq_len]
        Returns:
            out: loss value
        """
        # This is loss, so lower -> better
        loss = -self.si_snr(pred, target_audio)
        if self.return_dict:
            return {"loss": loss}
        return loss
