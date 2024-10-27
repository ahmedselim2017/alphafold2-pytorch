import torch
from alphafold.model.multi_head_attention import MultiHeadAttention


class MSARowAttentionWithPairBias(torch.nn.Module):
    """
    Implementation of the Algorithm 7.
    """

    def __init__(self, c_m: int, c_z: int, c=32, N_head=8):
        """
        Initializes MSARowAttentionWithPairBias.

        Args:
            c_m:    Embedding dimensions of the MSA representation.
            c_z:    Embedding dimensions of the pair representation.
            c:  Embedding dimension for multi-head attention.
            N_head: Number of heads for multi-head attention
        """

        super().__init__()

        self.layer_norm_m = torch.nn.LayerNorm(c_m)
        self.layer_norm_z = torch.nn.LayerNorm(c_z)
        self.linear_z = torch.nn.Linear(c_z, N_head, bias=False)

        self.mha = MultiHeadAttention(c_m, c, N_head, attn_dim=-2, gated=True)

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Algorithm 7.

        Args:
            m:  A PyTorch tensor of shape (*, N_seq, N_res, c_m) that contains
                the MSA representation.
            z:  A PyTorch tensor of shape (*, N_res, N_res, c_m) that contains
                the pair representation.

        Returns:
            Output PyTorch tensor with the shape of m
        """

        m = self.layer_norm_m(m)
        b = self.linear_z(self.layer_norm_z(z))

        # (*, N_res, N_res, N_head) -> (*, N_head, N_res, N_res)
        b = b.movedim(-1, -3)

        return self.mha(m, bias=b)


class MSAColumnAttention(torch.nn.Module):
    """
    Implementation of the Algorithm 8.
    """

    def __init__(self, c_m: int, c: int, N_head: int = 8):
        """
        Initializes MSAColumnAttention.

        Args:
            c_m:    Embedding dimensions of the MSA representation.
            c:  Embedding dimension for multi-head attention.
            N_head: Number of heads for multi-head attention
        """
        super().__init__()

        self.layer_norm_m = torch.nn.LayerNorm(c_m)
        self.mha = MultiHeadAttention(c_m, c, N_head, attn_dim=-3, gated=True)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Algorithm 8.

        Args:
            m:  A PyTorch tensor of shape (*, N_seq, N_res, c_m) that contains
                the MSA representation.

        Returns:
            Output PyTorch tensor with the shape of m
        """

        m = self.layer_norm_m(m)
        return self.mha(m)
