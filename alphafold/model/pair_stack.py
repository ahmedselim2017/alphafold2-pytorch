from alphafold.model.multi_head_attention import MultiHeadAttention
from alphafold.model.dropout import DropoutRowwise, DropoutColumnwise
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


class PairTransition(torch.nn.Module):
    """
    Implementation of the Algorithm 15.
    """

    def __init__(self, c_z: int, n: int = 4):
        """
        Initializes PairTransition.

        Args:
            c_z:    Embedding dimensions of the pair representation.
            n:  The factor that should be while expanding the original
                number of channels.
        """

        super().__init__()

        c_inter = c_z * n

        self.layer_norm = torch.nn.LayerNorm(c_z)
        self.linear_1 = torch.nn.Linear(c_z, c_inter)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(c_inter, c_z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Algorithm 15.

        Args:
            z:  A PyTorch tensor of shape (*, N_res, N_res, c_m) that contains
                the pairwise representation.

        Returns:
            Output PyTorch tensor with the shape of z
        """

        z = self.layer_norm(z)
        a = self.linear_1(z)
        z = self.linear_2(self.relu(a))

        return z


class PairStack(torch.nn.Module):
    """
    Implementation of the PairStack from the Algorithm 6.
    """

    def __init__(self, c_z: int):
        """
        Initializes the PairStack.

        Args:
            c_z:    Embedding dimensions of the pair representation.
        """

        super().__init__()

        self.dropout_rowwise = DropoutRowwise(p=0.25)
        self.dropout_columnwise = DropoutColumnwise(p=0.25)

        self.tri_mul_out = TriangleMultiplication(c_z, mult_type="outgoing")
        self.tri_mul_in = TriangleMultiplication(c_z, mult_type="incoming")

        self.tri_att_start = TriangleAttention(c_z, node_type="starting_node")
        self.tri_att_end = TriangleAttention(c_z, node_type="ending_node")

        self.pair_transition = PairTransition(c_z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PairStack module.

        Args:
            z:  A PyTorch tensor of shape (*, N_res, N_res, c_m) that contains
                the pairwise representation.

        Returns
            Output PyTorch tensor with the shape of z
        """

        z += self.dropout_rowwise(self.tri_mul_out(z))
        z += self.dropout_rowwise(self.tri_mul_in(z))
        z += self.dropout_rowwise(self.tri_att_start(z))
        z += self.dropout_columnwise(self.tri_att_end(z))
        z += self.pair_transition(z)

        return z
