import math
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    """
    Encoder layer that transforms initial audio
    into matrix of format similar to STFT.

    Args:
        n_freq: number of frequencies simulated
        kernel_size: parameter for inner 1d conv
        stride: parameter for inner 1d conv
    """

    def __init__(self, n_freq: int, kernel_size: int, stride: int):
        super().__init__()
        assert kernel_size % 4 == 0, "Only divided by 4 kernel_size are supported"
        self.Conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=n_freq,
            kernel_size=kernel_size,
            padding=kernel_size // 4,
            stride=stride,
            bias=False,
        )
        self.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, [batch_size, T]
        Returns:
            out: Tensor, [batch_size, n_freq, T']
        """
        x = self.Conv1d(x.unsqueeze(1))
        return self.ReLU(x)


class Decoder(nn.Module):
    """
    Encoder layer that transforms initial audio
    into matrix of format similar to STFT.

    Args:
        n_freq: number of frequencies simulated
        kernel_size: parameter for inner 1d conv
        stride: parameter for inner 1d conv
    """

    def __init__(self, n_freq: int, kernel_size: int, stride: int):
        super().__init__()
        assert kernel_size % 4 == 0, "Only divided by 4 kernel_size are supported"
        self.Conv1dTranspose = nn.ConvTranspose1d(
            in_channels=n_freq,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 4,
            stride=stride,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, [batch_size, n_speakers, n_freq, T']
        Returns:
            out: Tensor, [batch_size, n_speakers, T]
        """
        BS, NS, NF, T = x.shape
        out = self.Conv1dTranspose(x.view(BS * NS, NF, T)).squeeze(1)
        return out.view(BS, NS, -1)


class FeedForward(nn.Module):
    """
    FeedForward block in transformer layers

    Args:
        input_dim: size of input embeddings
        hidden_dim: size of embeddings in hidden layer
        dropout: dropout prbability
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()

        self.ffw = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, [..., input_dim]
        Returns:
            out: Tensor, [..., input_dim]
        """
        return self.ffw(x)


class TransformerBlock(nn.Module):
    """
    Both Intra and Inter transformer block.
    Same ideas as in original Transformer paper.

    Args:
        embed_dim: size of hidden state
        n_attention_heads: number of attetion heads in MHSA
        dropout: dropout probability
    """

    def __init__(self, embed_dim: int, n_attention_heads: int, dropout: float) -> None:
        super().__init__()

        self.pre_mhsa_ln = nn.LayerNorm(embed_dim)
        self.mhsa = nn.MultiheadAttention(
            embed_dim, n_attention_heads, dropout=dropout, batch_first=True
        )
        self.pos_mhsa_ln = nn.LayerNorm(embed_dim)
        self.ffw = FeedForward(embed_dim, embed_dim * 4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass assumes that positional encoding are beeing added into input

        Args:
            x: Tensor, [batch_size, time_dim, embed_dim]
        Returns:
            out: Tensor, [batch_size, time_dim, embed_dim]
        """
        y = self.pre_mhsa_ln(x)
        x = x + self.mhsa(y, y, y, attn_mask=None, key_padding_mask=None)[0]
        x = x + self.ffw(self.pos_mhsa_ln(x))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """
    Applies sinusoidal positional encoding to a tensor.
    At i, 2k: add sin(i / (10000^(2k / d_model)))
    At i, 2k + 1: add cos(i / (10000^(2k / d_model)))

    Args:
        hidden_dim: second dimension of tensor
        max_len: max lenght of tensor
    """

    def __init__(self, hidden_dim: int, max_len: int):
        super().__init__()

        positional_encoding = torch.zeros(max_len, hidden_dim)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(
            (math.log(10000.0) / hidden_dim)
            * torch.arange(0, hidden_dim, 2, dtype=torch.float)
        )

        positional_encoding[:, 0::2] = torch.sin(position / denominator)
        positional_encoding[:, 1::2] = torch.cos(position / denominator)
        
        positional_encoding = positional_encoding.unsqueeze(
            0
        )  # Add batch_size dimension
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, [batch_size, hidden_dim, input_len]
        Returns:
            out: Tensor, [batch_size, hidden_dim, input_len]
        """
        x = x + self.positional_encoding[:, :x.shape[1], :]
        return x


class SepFormerBlock(nn.Module):
    """
    Main block in SepFormer model. Computes attention inside
    each chopped block and between blocks as well.

    Args:
        n_freq: size of hidden state
        block_size: size of blocks in input tensor
        n_attention_heads: number of attention heads in MHSA
        n_intra_blocks: number of Intra blocks that models the short-term dependencies
        n_inter_blocks: number of Inter blocks that models the long-term dependencies
        dropout: dropout probability
    """

    def __init__(
        self,
        n_freq: int,
        block_size: int,
        n_attention_heads: int,
        n_intra_blocks: int,
        n_inter_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.intra_pos_encoding = SinusoidalPositionalEncoding(
            hidden_dim=n_freq, max_len=5000
        )
        self.intra_transformer = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=n_freq,
                    n_attention_heads=n_attention_heads,
                    dropout=dropout,
                )
                for _ in range(n_intra_blocks)
            ]
        )

        self.inter_pos_encoding = SinusoidalPositionalEncoding(
            hidden_dim=n_freq, max_len=5000
        )
        self.inter_transformer = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=n_freq,
                    n_attention_heads=n_attention_heads,
                    dropout=dropout,
                )
                for _ in range(n_inter_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, [batch_size, n_freq, block_size, n_blocks]
        Returns:
            out: Tensor, [batch_size, n_freq, block_size, n_blocks]
        """
        B, NF, BS, NB = x.shape

        # block size as time dim
        x = x.permute(0, 3, 2, 1).contiguous().view(NB * B, BS, NF)
        x_pos = self.intra_pos_encoding(x)
        x = x + self.intra_transformer(self.intra_pos_encoding(x))
        x = x.view(B, NB, BS, NF).contiguous().permute(0, 3, 2, 1)  # back to original shape

        # block amount as time dim
        x = x.permute(0, 2, 3, 1).contiguous().view(BS * B, NB, NF)
        x = x + self.inter_transformer(self.inter_pos_encoding(x))

        x = x.view(B, BS, NB, NF).permute(0, 3, 1, 2).contiguous()
        return x


class ChunkingMode(Enum):
    CHUNKING = 1
    UNCHUNKING = 2


class Chunking(nn.Module):
    """
    Chunking step of SepFormer model

    Args:
        block_size: size of blocks that sequence will be divided
    """

    def __init__(self, block_size: int) -> None:
        super().__init__()

        assert block_size % 2 == 0, "Only even block_size are supported"

        self.block_size = block_size
        self.block_stride = block_size // 2

    def forward(
        self, x: torch.Tensor, chunking_mode: ChunkingMode, n_padded: int | None = None
    ) -> tuple[torch.Tensor, int | None]:
        """
        Args:
            x: Tensor, [..., seq_len]
            chunking_mode: ChunkingMode, to make blocks or reverse to single sequence
            n_padded: only needed in ChunkingMode.UNCHUNKING
        Returns:
            out_1: Tensor, [..., block_size, n_blocks] if ChunkingMode.CHUNKING, else [..., seq_len]
            out_2: number of padding tokens added or None if ChunkingMode.UNCHUNKING
        """
        match chunking_mode:
            case ChunkingMode.CHUNKING:
                x_padded, n_padded = self.pad(x)
                out = torch.cat(
                    (
                        x_padded[..., self.block_stride :].view(
                            *x.shape[:-1], -1, self.block_size
                        ),
                        x_padded[..., : -self.block_stride].view(
                            *x.shape[:-1], -1, self.block_size
                        ),
                    ),
                    dim=-2,
                ).transpose(-1, -2)
                return out, n_padded
            case ChunkingMode.UNCHUNKING:
                assert (
                    n_padded is not None
                ), "n_padded should be not None when ChunkingMode.UNCHUNKING"
                n_blocks = x.shape[-1]

                left_part = (
                    x[..., : n_blocks // 2]
                    .transpose(-1, -2)
                    .view(*x.shape[:-2], -1)[..., : -(self.block_stride + n_padded)]
                )
                right_part = (
                    x[..., n_blocks // 2 :]
                    .transpose(-1, -2)
                    .view(*x.shape[:-2], -1)[..., self.block_stride : -n_padded]
                )
                original = (left_part + right_part) / 2
                return original, None

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Args:
            x: Tensor, [..., seq_len]
        Returns:
            out_1: Tensor, [..., padded_seq_len]
            out_2: number of padding tokens added
        """
        seq_len = x.shape[-1]
        to_add = (
            self.block_size
            - (self.block_stride + seq_len % self.block_size) % self.block_size
        )
        if to_add > 0:
            return (
                F.pad(x, (self.block_stride, to_add + self.block_stride), value=0.0),
                to_add,
            )
        return x, to_add


class SepFormerInner(nn.Module):
    """
    SepFormer inner blocks that works with encoded input

    Args:
        n_speakers: number of speaker in audio, also number of masks to predict
        n_freq: size of hidden state
        block_size: size of blocks that sequence will be divided
        n_sepformer_blocks: number of sequential SepFormerBlocks
        n_intra_blocks: number of Intra blocks inside SepFormerBlocks
        n_itner_blocks: number of Inter blocks inside SepFormerBlocks
        n_attention_heads: number of attention heads in MHSA
        dropout: dropout probability
    """

    def __init__(
        self,
        n_speakers: int,
        n_freq: int,
        block_size: int,
        n_sepformer_blocks: int,
        n_intra_blocks: int,
        n_inter_blocks: int,
        n_attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_speakers = n_speakers
        self.n_freq = n_freq

        self.ln_linear = nn.Sequential(nn.LayerNorm(n_freq), nn.Linear(n_freq, n_freq))
        self.chunking = Chunking(block_size)
        self.sepformer_blocks = nn.Sequential(
            *[
                SepFormerBlock(
                    n_freq=n_freq,
                    block_size=block_size,
                    n_attention_heads=n_attention_heads,
                    n_intra_blocks=n_intra_blocks,
                    n_inter_blocks=n_inter_blocks,
                    dropout=dropout,
                )
                for _ in range(n_sepformer_blocks)
            ]
        )

        self.post_sepformer_blocks = nn.Sequential(
            nn.PReLU(), nn.Linear(n_freq, n_speakers * n_freq)
        )
        self.final_ffw = nn.Sequential(
            FeedForward(n_freq, 4 * n_freq, dropout=dropout),
            FeedForward(n_freq, 4 * n_freq, dropout=dropout),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, [batch_size, n_freq, seq_len]
        Returns:
            out: Tensor, [batch_size, n_speakers, n_freq, seq_len]
        """
        # to [batch_size, seq_len, n_freq]
        x = x.permute(0, 2, 1)
        # back to original size
        x = self.ln_linear(x).permute(0, 2, 1)

        # [batch_size, n_freq, block_size, n_blocks]
        x, padded_size = self.chunking(x, chunking_mode=ChunkingMode.CHUNKING)

        x = self.sepformer_blocks(x)
        # [batch_size, n_freq * n_speakers, block_size, n_blocks]
        x = self.post_sepformer_blocks(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        # [batch_size, n_freq * n_speakers, seq_len]
        x, _ = self.chunking(
            x, chunking_mode=ChunkingMode.UNCHUNKING, n_padded=padded_size
        )

        # [batch_size, n_speakers, n_freq, seq_len]
        x = x.view(-1, self.n_speakers, self.n_freq, x.shape[-1])

        return self.final_ffw(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


class SepFormer(nn.Module):
    """
    SepFormer model from paper
    "Attention is all you need in speech separation"

    Args:
        n_speakers: number of speaker in audio, also number of masks to predict
        n_freq: size of hidden state
        kernel_size: kernel_size in encoder and decoder layers
        block_size: size of blocks that sequence will be divided
        n_sepformer_blocks: number of sequential SepFormerBlocks
        n_intra_blocks: number of Intra blocks inside SepFormerBlocks
        n_itner_blocks: number of Inter blocks inside SepFormerBlocks
        n_attention_heads: number of attention heads in MHSA
        dropout: dropout probability
    """

    def __init__(
        self,
        n_speakers: int,
        n_freq: int,
        kernel_size: int,
        block_size: int,
        n_sepformer_blocks: int,
        n_intra_blocks: int,
        n_inter_blocks: int,
        n_attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        assert kernel_size % 2 == 0, "Only even kernel_size are supported"
        self.encoder = Encoder(n_freq, kernel_size, kernel_size // 2)
        self.decoder = Decoder(n_freq, kernel_size, kernel_size // 2)
        self.sepformer_inner = SepFormerInner(
            n_speakers=n_speakers,
            n_freq=n_freq,
            block_size=block_size,
            n_sepformer_blocks=n_sepformer_blocks,
            n_intra_blocks=n_intra_blocks,
            n_inter_blocks=n_inter_blocks,
            n_attention_heads=n_attention_heads,
            dropout=dropout,
        )

    def forward(self, mix_audio: torch.Tensor, **batch):
        """
        Args:
            mix_audio: mixture of audio.
        Returns:
            output (dict): output dict containing logits.
        """
        x_encoded = self.encoder(mix_audio)
        masks = self.sepformer_inner(x_encoded)
        output = self.decoder(masks)
        return {"preds": output}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
