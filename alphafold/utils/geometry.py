import alphafold.utils.residue_constants as constants
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
        A PyTorch tensor with a shape of (*,4) that is the multiplication of q1
        and q2
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


def conjugate_quat(q: torch.Tensor) -> torch.Tensor:
    """
    Calculates the conjugate of a quaternion.

    Args:
        q:  A PyTorch tensor with a shape of (*, 4) that includes the
            quaternion

    Returns:
        A PyTorch tensor with a shape of (*, 4) that includes the conjugate
        of the give quaternion.
    """

    a = q[..., 0:1]
    v = q[..., 1:]

    return torch.cat((a, -v), dim=-1)


def quat_vector_mul(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Multiplies a vector v with a quaternion q as 'q * (0, v) * (~q)'.

    Args:
        q:  A PyTorch tensor with a shape of (*, 4) that includes the
            quaternion.
        v:  A PyTorch tensor with a shape of (*, 3) that includes the vector.

    Returns:
        A PyTorch tensor with a shape of (*, 3) that is the multiplication of
        q and v.
    """

    v_pad = torch.zeros(q.shape[:-1] + (1, ), device=v.device, dtype=v.dtype)

    # (*, 3) -> (* , 4)
    padded_v = torch.cat((v_pad, v), dim=-1)

    return quat_mul(q, quat_mul(padded_v, conjugate_quat(q)))[..., 1:]


def quat_to_3x3_rotation(q: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a 3x3 rotation matrix.

    Args:
        q:  A PyTorch tensor with a shape of (*, 4) that includes the
            quaternion.

    Returns:
        A PyTorch tensor with a shape of (*, 3, 3).
    """

    identity = torch.eye(3, dtype=q.dtype, device=q.device)
    identity = identity.broadcast_to(q.shape[:-1] + (3, 3))

    v1 = quat_vector_mul(q, identity[..., 0])
    v2 = quat_vector_mul(q, identity[..., 1])
    v3 = quat_vector_mul(q, identity[..., 2])

    return torch.stack((v1, v2, v3), dim=-1)


def assemble_4x4_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Assembles a 3x3 rotation matrix R and a transformation t to a 4x4
    homogenous matrix.

    Args:
        R:  A PyTorch tensor with a shape of (*, 3, 3).
        t:  A PyTorch tensor with a shape of (*, 3).

    Returns:
        A PyTorch tensor with a shape of (*, 4, 4).
    """

    Rt = torch.cat((R, t.unsqueeze(-1)), dim=-1)
    pad = torch.zeros(t.shape[:-1] + (1, 4), device=t.device, dtype=t.dtype)
    pad[..., -1] = 1

    return torch.cat((Rt, pad), dim=-2)


def warp_3d_point(T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Warps a 3D point through a homogenous 4x4 transform.

    Promotes the given point x to 4D by making it 1x4, applies the given 4x4
    transformation T to it, then converts the point to 3D.

    Args:
        T:  A PyTorch tensor with a shape of (*, 4, 4).
        x:  A PyTorch tensor with a ashape of (*, 3).

    Returns:
        A PyTorch tensor with a shape of (*, 3)
    """

    pad = torch.ones(x.shape[:-1] + (1, ), device=x.device, dtype=x.dtype)

    x = torch.cat((x, pad), dim=-1)

    x_warped = torch.einsum("...ij,...j->...i", T, x)

    return x_warped[..., :3]


def create_4x4_transform(ex: torch.Tensor, ey: torch.Tensor,
                         translation: torch.Tensor) -> torch.Tensor:
    """
    Creates a 4x4 transform by orthonormalizing the ex and ey vectors
    using Gram-Schmidt orthonormalization and using the given translation.

    Args:
        ex: A PyTorch tensor with a shape of (*, 3).
        ey: A PyTorch tensor with a shape of (*, 3).
        translation:    A PyTorch tensor with a shape of (*, 3).

    Returns:
        A PyTorch tensor with a shape of (*, 4, 4).
    """

    R = create_3x3_rotation(ex, ey)
    return assemble_4x4_transform(R, translation)


def invert_4x4_transform(T: torch.Tensor) -> torch.Tensor:
    """
    Inverts a 4x4 transform.

    The inverse of a transform (R,t) can be found as (R.T, -R.T @ t).

    Args:
        T:  A PyTorch tensor with a shape of (*, 4, 4).

    Returns:
        A PyTorch tensor with a shape of (*, 4, 4).
    """

    R = T[..., :3, :3]
    t = T[..., :3, 3]

    R_new = torch.transpose(R, -1, -2)
    t_new = torch.einsum("...ij,...j->...i", -R_new, t)

    return assemble_4x4_transform(R_new, t_new)


def makeRotX(phi: torch.Tensor) -> torch.Tensor:
    """
    Creates a 4x4 transform for rotation of phi around the x axis where
    phi is given by (cos(phi), sin(phi)).

    Args:
        phi:    A PyTorch tensor with a shape of (*, 2)

    Returns:
        A PyTorch tensor with a shape of (*, 4, 4).
    """

    phi_cos = phi[..., 0]
    phi_sin = phi[..., 1]

    R = torch.zeros(phi.shape[:-1] + (3, 3),
                    device=phi.device,
                    dtype=phi.dtype)

    R[..., 0, 0] = 1
    R[..., 1, 1] = phi_cos
    R[..., 1, 2] = -phi_sin
    R[..., 2, 1] = phi_sin
    R[..., 2, 2] = phi_cos

    t = torch.zeros(phi.shape[:-1] + (3, ), device=phi.device, dtype=phi.dtype)

    return assemble_4x4_transform(R, t)


def calculate_non_chi_transforms() -> torch.Tensor:
    """
    Calculates transforms for the local backbone frames:

    backbone_group: identity
    pre_omega_group:    identity
    phi_group:
        ex: CA (0, 0, 0) -> N
        ey: CA -> C
            (0, 0, 0) -> (1, 0, 0) [when normalized]
        t:  N
    psi_group:
        ex: CA -> C
        ey: N -> CA
        t:  C

    Returns:
        Stacked transforms with a shape of (20, 4, 4, 4) which include 4 4x4
        transforms included in dimensions 2 and 3 for 20 amino acids.
    """

    backbone_group = torch.eye(4).broadcast_to((20, 4, 4))
    pre_omega_group = torch.eye(4).broadcast_to((20, 4, 4))

    phi_group = torch.zeros((20, 4, 4))
    psi_group = torch.zeros((20, 4, 4))

    phi_ey = torch.tensor([1., 0., 0.])
    for i, atom_pos in enumerate(
            constants.rigid_group_atom_position_map.values()):
        phi_ex = atom_pos["N"] - atom_pos["CA"]
        phi_group[i, ...] = create_4x4_transform(phi_ex, phi_ey, atom_pos["N"])

        psi_ex = atom_pos["C"] - atom_pos["CA"]
        psi_ey = atom_pos["CA"] - atom_pos["N"]
        psi_group[i, ...] = create_4x4_transform(psi_ex, psi_ey, atom_pos["C"])

    return torch.stack((backbone_group, pre_omega_group, phi_group, psi_group),
                       dim=1)


def calculate_chi_transforms() -> torch.Tensor:
    """
    Calculates transforms for the side-chain frames:

    chi1:
        ex: CA -> #SC0
        ey: CA -> N
        t:  #SC0
    chi2:
        ex: #SC0 -> #SC1
        ey: #SC0 -> CA
        t:  #SC1
    chi3:
        ex: #SC1 -> #SC2
        ey: #SC1 -> #SC0
        t: #SC2
    chi4:
        ex: #SC2 -> #SC3
        ey: #SC2 -> #SC1
        t: #SC3

    If a chi angle is not preset, it is replaced with an identity transform.

    Returns:
        Stacked transforms with a shape of (20, 4, 4, 4) which include 4 4x4
        transforms included in dimensions 2 and 3 for 20 amino acids.
    """

    transforms = torch.zeros((20, 4, 4, 4))

    for i, (res, atom_pos) in enumerate(
            constants.rigid_group_atom_position_map.items()):
        for j in range(4):
            if not constants.chi_angles_mask[i][j]:
                transforms[i, j, ...] = torch.eye(4)
                continue

            atom = constants.chi_angles_chain[res][j]

            if j == 0:
                ex = atom_pos[atom] - atom_pos["CA"]  # type: ignore
                ey = atom_pos["N"] - atom_pos["CA"]  # type: ignore
            else:
                ex = atom_pos[atom]
                ey = torch.tensor([-1., 0., 0.])

            transforms[i, j, ...] = create_4x4_transform(
                ex, ey, atom_pos[atom])  # type: ignore
    return transforms


def precalculate_rigid_transforms() -> torch.Tensor:
    """
    Calculates the non-chi and chi transforms.

    Returns:
        A PyTorch tensor with a shape of (20, 8, 4, 4).
    """

    return torch.cat(
        (calculate_non_chi_transforms(), calculate_chi_transforms()), dim=1)


def compute_global_transforms(T: torch.Tensor, alpha: torch.Tensor,
                              F: torch.Tensor) -> torch.Tensor:
    """
    Calculates the global frames for each aminoacid  by applying the global
    transform T and rotation transforms in between the side chain frames.

    Implementation of first 10 lines of the Alogrithm 24.

    Args:
        T:  A PyTorch tensor with a shape of (N_res, 4, 4) that contains the
            global backbone transform for each aminoacid.
        alpha:  A PyTorch tensor with a shape of (N_res, 7, 2) that contains
                the cosine and sine values of the angles omega, phi, psi, chi1,
                chi2, chi3, and chi4 in order.
        F:  A PyTorch tensor with a shape of (N_res,) that contains the labels
            for aminoacids encoded as indices.

    Returns:
        A PyTorch tensor with a shape of (N_res, 8, 4, 4) that includes the
        global frames for each aminoacid.
    """

    alpha = alpha / torch.linalg.norm(alpha, dim=-1, keepdim=True)

    omega, phi, psi, chi1, chi2, chi3, chi4 = torch.unbind(alpha, -2)

    all_rigid_transforms = precalculate_rigid_transforms().to(device=T.device,
                                                              dtype=T.dtype)
    local_transforms = all_rigid_transforms[F]

    global_transforms = torch.zeros_like(local_transforms,
                                         device=T.device,
                                         dtype=T.dtype)
    global_transforms[..., 0, :, :] = T

    for i, angle in enumerate([omega, phi, psi, chi1]):
        global_transforms[..., i +1, :, :] = \
                T @ local_transforms[..., i + 1, :, :] @ makeRotX(angle)
    for i, angle in enumerate([chi2, chi3, chi4]):
        j = i + 5
        global_transforms[..., j, :, :] = \
                global_transforms[..., j - 1,:,:] @ local_transforms[...,j,:,:] @ makeRotX(angle)
    return global_transforms


def compute_all_atom_coordinates(
        T: torch.Tensor, alpha: torch.Tensor,
        F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of the Algorithm 24.

    Args:
        T:  A PyTorch tensor with a shape of (N_res, 4, 4) that contains the
            global backbone transform for each aminoacid.
        alpha:  A PyTorch tensor with a shape of (N_res, 7, 2) that contains
                the cosine and sine values of the angles omega, phi, psi, chi1,
                chi2, chi3, and chi4 in order.
        F:  A PyTorch tensor with a shape of (N_res,) that contains the labels
            for aminoacids encoded as indices.

    Returns:
        A PyTorch tensor with a shape (N_res, 37, 3) that includes the global
        positions of each atom in each aminoacid and a PyTorch tensor with a
        shape of (N_res, 37) that includes the atom mask.
    """

    device = T.device
    dtype = T.dtype

    global_transforms = compute_global_transforms(T, alpha, F)
    global_transforms = global_transforms.to(device=T.device, dtype=T.dtype)

    # (20, 37, 3)
    atom_local_positions = constants.atom_local_positions
    atom_local_positions = atom_local_positions.to(device=device, dtype=dtype)
    atom_local_positions = atom_local_positions[F]

    local_pos_pad = torch.ones(atom_local_positions.shape[:-1] + (1, ),
                               device=device,
                               dtype=dtype)
    # (20, 37, 4)
    padded_local_positions = torch.cat((atom_local_positions, local_pos_pad),
                                       dim=-1)

    # (20, 37)
    atom_frame_inds = constants.atom_frame_inds.to(device=device)
    atom_frame_inds = atom_frame_inds[F]

    diff_dim = global_transforms.ndim - atom_frame_inds.ndim

    atom_frame_inds = atom_frame_inds.reshape(atom_frame_inds.shape +
                                              (1, ) * diff_dim)

    atom_frame_inds = atom_frame_inds.broadcast_to(
        atom_frame_inds.shape[:-diff_dim] +
        global_transforms.shape[-diff_dim:])

    # (N_res, 8, 4, 4) -> (N_res, 37, 4, 4)
    atom_frames = torch.gather(global_transforms,
                               dim=-3,
                               index=atom_frame_inds)

    global_positions = torch.einsum("...ijk,...ik->...ij", atom_frames,
                                    padded_local_positions)
    global_positions = global_positions[..., :3]

    atom_mask = constants.atom_mask.to(device=device)[F]

    return global_positions, atom_mask
