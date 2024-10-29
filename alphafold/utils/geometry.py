import torch


def create_3x3_rotation(ex: torch.Tensor, ey: torch.Tensor):
    """
    Creates a rotation matrix by orthonormalizing the ex and ey vectors
    using Gram-Schmidt orthonormalization.

    Args:
        ex: A PyTorch tensor of shape (*, 3).
        ey: A PyTorch tensor of shape (*, 3).
    
    Returns:
        A PyTorch tensor of shape (*,3,3).
    """

    x = ex / torch.linalg.vector_norm(ex, dim=-1, keepdim=True)
    y = ey - torch.sum(x * ey, dim=-1, keepdim=True) * x
    y = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    z = torch.linalg.cross(x, y, dim=-1)

    return torch.stack((x, y, z), dim=-1)
