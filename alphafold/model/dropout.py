import torch


class SharedDropout(torch.nn.Module):
    """
    A Dropout module that is shared for one dimension.
    """

    def __init__(self, shared_dim: int, p: float):
        """
        Initializes SharedDropout.

        Args:
            shared_dim: The shared dimension that should be used.
            p:  Dropout probability.
        """

        super().__init__()

        self.dropout = torch.nn.Dropout(p)
        self.shared_dim = shared_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SharedDropout.

        Args:
            x:  The PyTorch tensor that should be used.

        Returns:
            A PyTorch tensor with the shape of x.
        """

        mask_shape = list(x.shape)
        mask_shape[self.shared_dim] = 1
        mask = torch.ones(mask_shape, device=x.device)

        return x * self.dropout(mask)


class DropoutRowwise(SharedDropout):
    """
    Row-wise shared dropout module.
    """

    def __init__(self, p: float):
        """
        Initializes DropoutRowwise.

        Args:
            p:  Dropout probability.
        """

        super().__init__(shared_dim=-2, p=p)


class DropoutColumnwise(SharedDropout):
    """
    Column-wise shared dropout module.
    """

    def __init__(self, p: float):
        """
        Initializes DropoutRowwise.

        Args:
            p:  Dropout probability.
        """

        super().__init__(shared_dim=-3, p=p)
