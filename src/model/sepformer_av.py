import math
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn

from src.model.sepformer import Encoder as AudioEncoder
from src.model.sepformer import Decoder as AudioDecoder
from src.model.sepformer import FeedForward, TransformerBlock, SinusoidalPositionalEncoding
from src.model.sepformer import ChunkingMode, Chunking

class VideoEncoder(nn.Module):
    """
    VideoEncoder

    Args:
        input_dim: input embedding size
        output_dim: output embedding size
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim, bias=False),
            nn.PReLU(),
            nn.BatchNorm1d(input_dim),
        )

        self.downsample = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: Tensor, [batch_size, input_dim, seq_len]
        Returns:
            out: Tensor, [batch_size, output_dim, seq_len]
        """
        video = self.encoder(video) + video
        return self.downsample(video.transpose(1, 2)).transpose(1, 2)


class CrossModalAttention(nn.Module):
    """
    CrossModalAttention block for attention between audio and video, just the model attention part

    Args:
        embed_dim: size of hidden state
        n_attention_heads: number of attetion heads in MHSA
        dropout: dropout probability
    """

    def __init__(
        self, embed_dim: int, n_attention_heads: int, dropout: float
    ) -> None:
        super().__init__()
        assert embed_dim % n_attention_heads == 0
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=False)
        self.n_head = n_attention_heads

        self.dropout = dropout

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Tensor, [batch_size, block_size, block_count, embed_dim]
            video: Tensor, [batch_size, video_lenght, embed_dim]
        Returns:
            out: Tensor, [batch_size, block_size, block_count, embed_dim]
        Expecting video_lenght == block_count
        """
        B, R, T, C = audio.shape
        q, k, v = self.wq(video), self.wk(audio), self.wv(audio)
        k = k.view(B, R, T, self.n_head, C // self.n_head).transpose(2, 3)
        v = v.view(B, R, T, self.n_head, C // self.n_head).transpose(2, 3) 
        q = q.view(B, 1, T, self.n_head, C // self.n_head).transpose(2, 3)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(2, 3).contiguous().view(B, R, T, C)
        out = self.wo(out)
        return out


class CrossModelAttentionBlock(nn.Module):
    """
    CrossModalAttention block for attention between audio and video

    Args:
        embed_dim: size of hidden state
        hidden_dim: size of hidden state in FeedForward
        n_attention_heads: number of attetion heads in MHSA
        dropout: dropout probability
    """

    def __init__(
        self, embed_dim: int, hidden_dim: int, n_attention_heads: int, dropout: float
    ) -> None:
        super().__init__()

        self.pre_mhsa_ln = nn.LayerNorm(embed_dim)
        self.cross_attention = CrossModalAttention(
            embed_dim, n_attention_heads, dropout=dropout
        )
        self.pos_mhsa_ln = nn.LayerNorm(embed_dim)
        self.ffw = FeedForward(embed_dim, hidden_dim, dropout=dropout)

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Tensor, [batch_size, block_size, block_count, embed_dim]
            video: Tensor, [batch_size, video_lenght, embed_dim]
        Returns:
            out: Tensor, [batch_size, block_size, block_count, embed_dim]
        Expecting video_lenght == block_count
        """
        audio_normed = self.pre_mhsa_ln(audio)
        audio = audio + self.cross_attention(audio_normed, video)
        audio = audio + video.unsqueeze(1) + self.ffw(self.pos_mhsa_ln(audio))
        return audio


class AVSepFormerBlock(nn.Module):
    """
    Main block in SepFormer model. Computes attention inside
    each chopped block and between blocks as well.

    Args:
        n_freq: size of hidden state
        n_hidden_ffw: size if hidden state in FeedForward layer reverse bottleneck
        block_size: size of blocks in input tensor
        n_attention_heads: number of attention heads in MHSA
        n_intra_blocks: number of Intra blocks that models the short-term dependencies
        n_inter_blocks: number of Inter blocks that models the long-term dependencies
        dropout: dropout probability
    """

    def __init__(
        self,
        n_freq: int,
        n_hidden_ffw: int,
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
                    hidden_dim=n_hidden_ffw,
                    n_attention_heads=n_attention_heads,
                    dropout=dropout,
                )
                for _ in range(n_intra_blocks)
            ]
        )

        self.video_pos_encoding = SinusoidalPositionalEncoding(
            hidden_dim=n_freq, max_len=5000
        )
        self.cross_modal_transformer = CrossModelAttentionBlock(
            embed_dim=n_freq,
            hidden_dim=n_hidden_ffw,
            n_attention_heads=n_attention_heads,
            dropout=dropout,
        )

        self.inter_pos_encoding = SinusoidalPositionalEncoding(
            hidden_dim=n_freq, max_len=5000
        )
        self.inter_transformer = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=n_freq,
                    hidden_dim=n_hidden_ffw,
                    n_attention_heads=n_attention_heads,
                    dropout=dropout,
                )
                for _ in range(n_inter_blocks)
            ]
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Tensor, [batch_size, n_freq, block_size, n_blocks]
            video: Tensor, [batch_size, video_lenght = n_blocks, n_freq]
        Returns:
            out: Tensor, [batch_size, n_freq, block_size, n_blocks]
        """
        x = audio
        B, NF, BS, NB = x.shape

        # block size as time dim
        x = x.permute(0, 3, 2, 1).contiguous().view(NB * B, BS, NF)
        x = x + self.intra_transformer(self.intra_pos_encoding(x))
        x = (
            x.view(B, NB, BS, NF).contiguous().permute(0, 3, 2, 1)
        )  # back to original shape

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.cross_modal_transformer(x, video)

        # block amount as time dim
        x = x.view(BS * B, NB, NF)
        x = x + self.inter_transformer(self.inter_pos_encoding(x))

        x = x.view(B, BS, NB, NF).permute(0, 3, 1, 2).contiguous()
        return x


class AVSepFormerInner(nn.Module):
    """
    SepFormer inner blocks that works with encoded input

    Args:
        n_freq: size of hidden state
        n_hidden_ffw: size if hidden state in FeedForward layer reverse bottleneck
        block_size: size of blocks that sequence will be divided
        n_sepformer_blocks: number of sequential SepFormerBlocks
        n_intra_blocks: number of Intra blocks inside SepFormerBlocks
        n_itner_blocks: number of Inter blocks inside SepFormerBlocks
        n_attention_heads: number of attention heads in MHSA
        dropout: dropout probability
    """

    def __init__(
        self,
        n_freq: int,
        n_hidden_ffw: int,
        block_size: int,
        n_sepformer_blocks: int,
        n_intra_blocks: int,
        n_inter_blocks: int,
        n_attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_freq = n_freq

        self.ln_linear = nn.Sequential(nn.LayerNorm(n_freq), nn.Linear(n_freq, n_freq))
        self.chunking = Chunking(block_size)
        self.av_sepformer_blocks = nn.ModuleList(
            [
                AVSepFormerBlock(
                    n_freq=n_freq,
                    n_hidden_ffw=n_hidden_ffw,
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
            nn.PReLU(), nn.Linear(n_freq, n_freq)
        )
        self.final_ffw = nn.Sequential(
            FeedForward(n_freq, n_hidden_ffw, dropout=dropout),
            FeedForward(n_freq, n_hidden_ffw, dropout=dropout),
            nn.ReLU(),
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Tensor, [batch_size, n_freq, seq_len]
            video: Tensor, [batch_size, n_freq, video_len]
        Returns:
            out: Tensor, [batch_size, n_freq, seq_len]
        """
        x = audio
        # to [batch_size, seq_len, n_freq]
        x = x.permute(0, 2, 1)
        # back to original size
        x = self.ln_linear(x).permute(0, 2, 1)

        # [batch_size, n_freq, block_size, n_blocks]
        x, padded_size = self.chunking(x, chunking_mode=ChunkingMode.CHUNKING)

        for block in self.av_sepformer_blocks:
            x = block(x, video.transpose(1, 2))
        # [batch_size, n_speakers, block_size, n_blocks]
        x = self.post_sepformer_blocks(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        # [batch_size, n_freq, seq_len]
        x, _ = self.chunking(
            x, chunking_mode=ChunkingMode.UNCHUNKING, n_padded=padded_size
        )

        return self.final_ffw(x.permute(0, 2, 1)).permute(0, 2, 1)


class AVSepFormer(nn.Module):
    """
    AV-SepFormer model from paper
    https://arxiv.org/pdf/2306.14170
    Args:
        n_freq: size of hidden state
        n_video_embed: input size of video embedding
        n_hidden_ffw: size if hidden state in FeedForward layer reverse bottleneck
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
        n_freq: int,
        n_video_embed: int,
        n_hidden_ffw: int,
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
        self.audio_encoder = AudioEncoder(n_freq, kernel_size, kernel_size // 2)
        self.audio_decoder = AudioDecoder(n_freq, kernel_size, kernel_size // 2)
        self.video_encoder = VideoEncoder(n_video_embed, n_freq)
        self.av_sepformer_inner = AVSepFormerInner(
            n_freq=n_freq,
            n_hidden_ffw=n_hidden_ffw,
            block_size=block_size,
            n_sepformer_blocks=n_sepformer_blocks,
            n_intra_blocks=n_intra_blocks,
            n_inter_blocks=n_inter_blocks,
            n_attention_heads=n_attention_heads,
            dropout=dropout,
        )

    def forward(self, mix_audio: torch.Tensor, target_video: torch.Tensor, **batch):
        """
        Args:
            mix_audio: mixture of audio.
            target_video: video embeddings
        Returns:
            output (dict): output dict containing logits.
        """
        video_encoded = self.video_encoder(target_video.transpose(1, 2))
        audio_encoded = self.audio_encoder(mix_audio)
        masks = self.av_sepformer_inner(audio_encoded, video_encoded)
        predicted = (audio_encoded * masks).unsqueeze(1)
        output = self.audio_decoder(predicted).squeeze(1)
        return {"pred": output}

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
