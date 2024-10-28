import torch


class RecyclingEmbedder(torch.nn.Module):
    """
    Implementation of the Algorithm 32.
    """

    def __init__(self, c_m: int, c_z: int):
        """
        Initializes the recycling embedder.

        Args:
            c_m:    Embedding dimensions of the MSA representation.
            c_z:    Embedding dimensions of the pair representation.
        """

        super().__init__()

        self.bin_start = 3.25
        self.bin_end = 20.75
        self.bin_count = 15

        self.linear = torch.nn.Linear(self.bin_count, c_z)
        self.layer_norm_m = torch.nn.LayerNorm(c_m)
        self.layer_norm_z = torch.nn.LayerNorm(c_z)

    def forward(self, m_prev: torch.Tensor, z_prev: torch.Tensor,
                x_prev: torch.Tensor):
        """
        Forward pass for the recycling embedder. The residues that are below
        the smallest class are not assigned to a class similar to the OpenFold
        contrary to the supplementary information.

        Args:
            m_prev: MSA representation of previous iteration with a shape of
                    (*, N_seq, N_res, c_m).
            z_prev: MSA representation of previous iteration with a shape of
                    (*, N_res, N_res, c_m).
            x_prev: Pseudo-beta positions from the previous iterations with a
                    shape of (*, N_res, 3). The C-alpha position is used for
                    glycin as it does not have a C-beta atom.

        Returns:
            PyTorch tensors with a shape of (*, N_res, c_m) and
            (*, N_res, N_res, c_z).
        """

        # (*, N_res, N_res)
        d = torch.linalg.vector_norm(x_prev.unsqueeze(-2) -
                                     x_prev.unsqueeze(-3),
                                     dim=-1)

        bins_lower = torch.linspace(self.bin_start,
                                    self.bin_end,
                                    self.bin_count,
                                    device=x_prev.device)
        bins_upper = torch.cat(
            [bins_lower[1:],
             torch.tensor([torch.inf], device=x_prev.device)])

        d = d.unsqueeze(-1)
        d = ((d > bins_lower) * (d < bins_upper)).type(x_prev.dtype)

        d = self.linear(d)

        z = d + self.layer_norm_z(z_prev)
        m = self.layer_norm_m(m_prev[..., 0, :, :])

        return m, z
