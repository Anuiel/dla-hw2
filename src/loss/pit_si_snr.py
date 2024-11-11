import itertools

import torch
from torch import nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class PIT_SISNR(nn.Module):
    """
    Permutation invariant loss for SI-SNR
    """

    def __init__(self, n_speakers: int, return_dict: bool = True):
        super().__init__()
        self.return_dict = return_dict
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.n_speakers = n_speakers

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, **batch
    ) -> dict | float:
        """
        Loss function calculation logic.

        Args:
            preds: Tensor, [batch_size, n_speakers, seq_len]
            targets: Tensor, [batch_size, n_speakers, seq_len]
        Returns:
            out: loss value
        """
        # This is loss, so lower -> better
        loss = (
            -sum(
                max(
                    [
                        self.si_snr(pred[p, :], target)
                        for p in itertools.permutations(range(self.n_speakers))
                    ]
                )
                for pred, target in zip(preds, targets)
            )
            / preds.shape[0]
        )
        if self.return_dict:
            return {"loss": loss}
        return loss
