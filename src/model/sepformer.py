import torch
from torch import nn
from torch.nn import Sequential


class Encoder(nn.Module): 
    """
    Encoder layer that transforms initial audio
    into matrix of format similar to STFT. 
    """

    def __init__(self, n_freq, kernel_size, stride):
        """
        Args: 
            n_freq: number of frequencies simulated
            kernel_size: parameter for inner 1d conv
            stride: parameter for inner 1d conv
        """

        super().__init__()

        self.Conv1d = nn.Conv1d(in_channels=1,
                                out_channels=n_freq,
                                kernel_size=kernel_size,
                                stride=stride,
                                bias=False)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        """
        Args: 
            x: Tensor, [batch_size, T]
        Returns: 
            Tensor, [batch_size, n_freq, T']
        """
        
        x = self.Conv1d(x)
        return self.ReLU(x)


class Decoder(nn.Module): 
    """
    Decoder layer that transforms matrix of format
    similar to STFT to to original audio format.
    """

    def __init__(self, n_freq, kernel_size, stride):
        """
        Args: 
            n_freq: number of frequencies simulated
            kernel_size: parameter for inner 1d conv
            stride: parameter for inner 1d conv

        Returns: 
            Tensor: [n_freq, T']
        """

        super().__init__()

        # Use transpose version so output dimension match original audio len
        self.ConvTranspose1d = nn.ConvTranspose1d(in_channels=n_freq,
                                out_channels=1,
                                kernel_size=kernel_size,
                                stride=stride,
                                bias=False) 

        self.ReLU = nn.ReLU()

    def forward(self, x):
        """
        Args: 
            x: Tensor, [batch_size, n_freq, T']

        Returns: 
            Tensor, [batch_size, T]
        """
        x = self.ConvTranspose1d(x)
        return self.ReLU(x)


class TransformerBlock(nn.Module):
    """
    Both Intra and Inter transformer block. 
    Same ideas as in original Transformer paper. 
    """

    def __init__(self, embed_dim, n_attention_heads, dropout_p):
        self.layer_norm_input = nn.LayerNorm(normalized_shape=embed_dim)
        self.multihead_attention = nn.modules.activation.MultiheadAttention(embed_dim, n_attention_heads, dropout=dropout_p)
        self.layer_norm_attention = nn.LayerNorm(normalized_shape=embed_dim)
        self.feed_forward = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout_p),
                                         nn.Linear(embed_dim * 4, embed_dim))
        self.dropout_final = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x_layer = self.layer_norm_input(x) 
        x_attention = self.layer_norm_attention(self.multihead_attention(x_layer) + x)
        x_feedforward = self.dropout_final(self.feed_forward(x_attention)) + x
        return x_feedforward
    

class SinusoidalPositionalEncoding(nn.Module):
    """
    Applies sinusoidal positional encoding to a tensor. 
    At i, 2k: add sin(i / (10000^(2k / d_model)))
    At i, 2k + 1: add cos(i / (10000^(2k / d_model))) 
    """

    def __init__(self, hidden_dim, input_len):
        """
        Args:
            hidden_dim: second dimension of tensor
            input_len: third dimension of tensor
        """
        super().__init__()

        positional_encoding = torch.zeros(input_len, hidden_dim)
        
        position = torch.arange(0, input_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp((torch.log(10000.0) / hidden_dim) * torch.arange(0, hidden_dim, 2, dtype=torch.float))

        positional_encoding[:, 0::2] = torch.sin(position / denominator)
        positional_encoding[:, 1::2] = torch.cos(position / denominator)

        positional_encoding = positional_encoding.unsqueeze(0) # Add batch_size dimension

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, hidden_dim, input_len]
        """
        x = x + self.positional_encoding[:, :, :x.shape[2]]
        return x


class SepFormerBlock(nn.Module):
    """ 
    Main block in SepFormer model. Computes attention inside 
    each chopped block and between blocks as well. 
    """

    def __init__(self, n_freq, chop_block_len, n_attention_heads, n_intra_blocks, n_inter_blocks, dropout):
        super().__init__()

        self.n_intra_blocks = n_intra_blocks
        self.n_inter_blocks = n_inter_blocks

        self.intra_positional_encoding = SinusoidalPositionalEncoding(hidden_dim=n_freq, input_len=chop_block_len)


        self.intra_transformer = nn.ModuleList([])
        for i in range(self.n_intra_blocks):
            self.intra_transformer.append(TransformerBlock(embed_dim=chop_block_len,
                                                                  n_attention_heads=n_attention_heads,
                                                                  dropout=dropout))

        
        self.inter_positional_encoding = SinusoidalPositionalEncoding(hidden_dim=chop_block_len, input_len=n_freq)
        self.inter_transformer = nn.ModuleList([])
        for i in range(self.n_inter_blocks):
            self.inter_transformer.append(TransformerBlock(embed_dim=n_freq,
                                                                  nhead=n_attention_heads,
                                                                  dropout=dropout))

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, F - n_freq, C - chop block len, N_C - chop blocks]
        """
        pass
    
class SepFormer(nn.Module):
    """
    SepFormer model from paper 
    "Attention is all you need in speech separation"
    """

    def __init__(self, n_feats, n_class, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        

    def forward(self, x, **batch):
        """
        Model forward method.

        Args:
            x (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        # Encoder
        x_encoded = Encoder(x) 


        # Decoder is called 2 times for each source
        #output = Decoder(x)


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
