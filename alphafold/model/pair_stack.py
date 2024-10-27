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
