import torch


def create_3x3_rotation(ex: torch.Tensor, ey: torch.Tensor) -> torch.Tensor:
    """
    Creates a rotation matrix by orthonormalizing the ex and ey vectors
    using Gram-Schmidt orthonormalization.

    Args:
        ex: A PyTorch tensor with a shape of (*, 3).
        ey: A PyTorch tensor with a shape of (*, 3).

    Returns:
        A PyTorch tensor of shape (*,3,3).
    """

    x = ex / torch.linalg.vector_norm(ex, dim=-1, keepdim=True)
    y = ey - torch.sum(x * ey, dim=-1, keepdim=True) * x
    y = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    z = torch.linalg.cross(x, y, dim=-1)

    return torch.stack((x, y, z), dim=-1)


def quat_from_axis(phi: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """
    Creates a quaternion with a scalar cos(phi/2) and a vector sin(phi/2)*n

    Args:
        phi:    A PyTorch tensor with a shape of (*,).
        n:  A PyTorch tensor with a shape of (*, 3).

    Returns:
        A PyTorch tensor with a shape of (*, 4)
    """

    a = torch.cos(phi / 2).unsqueeze(-1)
    v = torch.sin(phi / 2).unsqueeze(-1) * n

    return torch.cat((a, v), dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiplies the given quaternions.

    Args:
        q1: A PyTorch tensor with a shape of (*,4) that includes the first
            quaternion
        q2: A PyTorch tensor with a shape of (*,4) that includes the second
            quaternion
    Returns:
        A PyTorch tensor of shape (*,4) that is the multiplication of q1 and q2
    """

    # q3 = (c,u)
    # q3 = q1*q2 = (a,v) * (b,w) = (ab - v dot w, aw + bv + v cross w)

    a = q1[..., 0:1]  # (*, 1)
    v = q1[..., 1:]  # (*, 3)

    b = q2[..., 0:1]  # (*, 1)
    w = q2[..., 1:]  # (*, 3)

    c = a * b - torch.sum(v * w, dim=-1, keepdim=True)
    u = a * w + b * v + torch.linalg.cross(v, w, dim=-1)

    return torch.cat((c, u), dim=-1)
