import torch


class StructureModuleTransition(torch.nn.Module):
    """
    Implementation of the lines 8 and 9 in the Algorithm 20.
    """

    def __init__(self, c_s: int):
        """
        Initializes the StructureModuleTransition.

        Args:
            c_s:    Dimension of the single representation.
        """

        super().__init__()

        self.c_s = c_s

        self.linear_1 = torch.nn.Linear(c_s, c_s)
        self.linear_2 = torch.nn.Linear(c_s, c_s)
        self.linear_3 = torch.nn.Linear(c_s, c_s)
        self.layer_norm = torch.nn.LayerNorm(c_s)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.ReLU()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the StructureModuleTransition.

        Args:
            s:  A PyTorch tensor with a shape of (*, N_res, c_s) that includes
                the single representation.

        Returns:
            A PyTorch tensor with the same shape as s.
        """

        s = s + self.linear_3(
            self.relu(self.linear_2(self.relu(self.linear_1(s)))))
        s = self.layer_norm(self.dropout(s))
        return s
