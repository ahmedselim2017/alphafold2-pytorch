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

    def __init__(self, c_m: int, c: int = 32, N_head: int = 8):
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


class MSATransition(torch.nn.Module):
    """
    Implementation of the Algorithm 9.
    """

    def __init__(self, c_m: int, n: int = 4):
        """
        Initializes  MSATransition.

        Args:
            c_m:    Embedding dimensions of the MSA representation.
            n:  The factor that should be while expanding the original
                number of channels.
        """

        super().__init__()

        c_inter = c_m * n

        self.layer_norm = torch.nn.LayerNorm(c_m)
        self.linear_1 = torch.nn.Linear(c_m, c_inter)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(c_inter, c_m)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Algorithm 9.

        Args:
            m:  A PyTorch tensor of shape (*, N_seq, N_res, c_m) that contains
                the MSA representation.

        Returns:
            Output PyTorch tensor with the shape of m
        """

        m = self.layer_norm(m)
        a = self.linear_1(m)
        m = self.linear_2(self.relu(a))
        return m


class OuterProductMean(torch.nn.Module):
    """
    Implementation of the Algorithm 10.
    """

    def __init__(self, c_m: int, c_z: int, c: int = 32):
        """
        Initializes OuterProductMean.

        Args:
            c_m:    Embedding dimensions of the MSA representation.
            c_z:    Embedding dimensions of the pair representation.
            c:  Embedding dimension for a and b.
        """

        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(c_m)
        self.linear_1 = torch.nn.Linear(c_m, c)
        self.linear_2 = torch.nn.Linear(c_m, c)
        self.linear_out = torch.nn.Linear(c * c, c_z)

        return

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Algorithm 10.

        The Supplementary Information of AlphaFold2 paper calculates mean
        before flattening whereas the DeepMind's AlphaFold2 implementation
        calculates sums, then flattens and finally divides the output tensor
        to the N_seq. The Algorithm 10 is implemented in the same way as the
        DeepMind's AlphaFold2 implementation.

        Args:
            m:  A PyTorch tensor of shape (*, N_seq, N_res, c_m) that contains
                the MSA representation.

        Returns:
            Output PyTorch tensor with the shape of (*, N_res, N_res, c_z)
        """

        N_seq = m.shape[-3]

        m = self.layer_norm(m)

        # (*, N_seq, N_res, c_m) -> (*, N_seq, N_res, c)
        a = self.linear_1(m)

        # (*, N_seq, N_res, c_m) -> (*, N_seq, N_res, c)
        b = self.linear_2(m)

        # (*, N_res, N_res, c, c)
        o = torch.einsum("...sic,...sjk->...ijck", a, b)

        # (*, N_res, N_res, c, c) -> (*, N_res, N_res, c**2)
        o = torch.flatten(o, start_dim=-2)

        # (*, N_res, N_res, c**2) -> (*, N_res, N_res, c_z)
        z = self.linear_out(o) / N_seq

        return z
