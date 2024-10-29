import torch
import math

control_folder = "./control_values/geometry"


def test_create_3x3_rotatiton():
    from alphafold.utils.geometry import create_3x3_rotation

    ex = torch.tensor([-0.6610, -0.7191, 0.1293])
    ey = torch.tensor([-0.2309, 1.7710, -1.3062])
    ex_batch = ex.broadcast_to(5, 3)
    ey_batch = ey.broadcast_to(5, 3)

    R = create_3x3_rotation(ex, ey)
    R_batch = create_3x3_rotation(ex_batch, ey_batch)

    R_exp = torch.tensor([[-0.6709, -0.6218,
                           0.4041], [-0.7299, 0.4572, -0.5082],
                          [0.1312, -0.6359, -0.7605]])
    R_exp_batch = R_exp.broadcast_to((5, 3, 3))

    assert torch.allclose(R, R_exp, atol=1e-3), 'Error in single computation.'
    assert torch.allclose(R_batch, R_exp_batch,
                          atol=1e-3), 'Error in batched computation.'


def test_quat_mul():
    from alphafold.utils.geometry import quat_from_axis, quat_mul

    phi = torch.tensor(math.pi / 4)
    n = torch.tensor([0.2, 0.5, -0.3])
    n = n / torch.linalg.vector_norm(n)

    phi_batch = phi.broadcast_to(2, 5)
    n_batch = n.broadcast_to(2, 5, 3)

    q = quat_from_axis(phi, n)
    q_batch = quat_from_axis(phi_batch, n_batch)

    q_exp = torch.load(f'{control_folder}/quat_from_axis_check.pt')

    assert torch.allclose(q, q_exp,
                          atol=1e-5), 'Error in quat_from_axis, single use.'
    assert torch.allclose(q_batch, q_exp.broadcast_to((2, 5, 4)),
                          atol=1e-5), 'Error in quat_from_axis, batched use.'

    p = torch.tensor([0.3, -0.4, 0.1, 0.8])
    p_batch = p.broadcast_to(2, 5, 4)

    pq = quat_mul(p, q)
    pq_batch = quat_mul(p_batch, q_batch)

    pq_exp = torch.load(f'{control_folder}/quat_mul_check.pt')

    assert torch.allclose(pq, pq_exp,
                          atol=1e-5), 'Error in quat_mul, single use.'
    assert torch.allclose(pq_batch, pq_exp.broadcast_to((2, 5, 4)),
                          atol=1e-5), 'Error in quat_mul, batched use.'
