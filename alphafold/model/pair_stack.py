from alphafold.model.multi_head_attention import MultiHeadAttention
import torch


class TriangleMultiplication(torch.nn.Module):
    """
    Implementation of the Algorithm 11 and Algorithm 12.
    """

    def __init__(self, c_z: int, mult_type: str, c=128):
        """
        Initializes TriangleMultiplication.

        Args:
            c_z:    Embedding dimensions of the pair representation.
            mutl_type:  Type of the triangle multiplication. Either 'outgoing'
                or 'incoming'.
            c:  Embedding dimension.
        """

        super().__init__()

        if mult_type not in ["outgoing", "incoming"]:
            raise ValueError(
                ("The mult_type must be either 'outgoing' or"
                 f"'incoming'. The given mult_type is: {mult_type}"))
        self.mult_type = mult_type

        self.layer_norm_in = torch.nn.LayerNorm(c_z)
        self.linear_a_p = torch.nn.Linear(c_z, c)
        self.linear_a_g = torch.nn.Linear(c_z, c)
        self.linear_b_p = torch.nn.Linear(c_z, c)
        self.linear_b_g = torch.nn.Linear(c_z, c)
        self.linear_g = torch.nn.Linear(c_z, c_z)
        self.linear_z = torch.nn.Linear(c, c_z)
        self.layer_norm_out = torch.nn.LayerNorm(c)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Algorithm 11 and Algoritm 12.

        Args:
            z:  A PyTorch tensor of shape (*, N_res, N_res, c_z) that contains
                the pairwise representation.

        Returns:
            Output PyTorch tensor with the shape of z.
        """

        z = self.layer_norm_in(z)
        a = torch.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = torch.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)
        g = torch.sigmoid(self.linear_g(z))

        if self.mult_type == "outgoing":
            z = torch.einsum("...ikc,...jkc->...ijc", a, b)
        else:
            z = torch.einsum("...kic,...kjc->...ijc", a, b)

        out = g * self.linear_z(self.layer_norm_out(z))
        return out


class TriangleAttention(torch.nn.Module):
    """
    Implementation of the Algorithm 13 and 14.
    """

    def __init__(self, c_z: int, node_type: str, c: int = 32, N_head: int = 4):
        """
        Initializes TriangleAttention.

        Args:
            c_z:    Embedding dimensions of the pair representation.
            node_type:  Type of the triangle attention. Either 'starting_node'
                or 'ending_node'.
            c:  Embedding dimension.
            N_head: Number of heads for multi head attention.
        """

        super().__init__()

        if node_type not in ["starting_node", "ending_node"]:
            raise ValueError(
                ("The node_type must be either 'starting_node' or"
                 f"'ending_node'. The given node_type is: {node_type}"))

        self.node_type = node_type
        if node_type == "starting_node":
            attn_dim = -2
        else:
            attn_dim = -3

        self.layer_norm = torch.nn.LayerNorm(c_z)
        self.mha = MultiHeadAttention(c_z,
                                      c,
                                      N_head,
                                      attn_dim=attn_dim,
                                      gated=True)
        self.linear = torch.nn.Linear(c_z, N_head, bias=False)

        if node_type == 'starting_node':
            attn_dim = -2
        else:
            attn_dim = -3

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Algorithm 13 and Algoritm 14.

        Args:
            z:  A PyTorch tensor of shape (*, N_res, N_res, c_z) that contains
                the pairwise representation.

        Returns:
            Output PyTorch tensor with the shape of z.
        """

        z = self.layer_norm(z)

        # (*, N_res, N_res, c_z) -> (*, N_res, N_res, N_head)
        b = self.linear(z)
        # (*, N_res, N_res, N_head) -> (*, N_head, N_res, N_res)
        b = b.movedim(-1, -3)

        if self.node_type == "ending_node":
            b = b.transpose(-1, -2)

        return self.mha(z, bias=b)
