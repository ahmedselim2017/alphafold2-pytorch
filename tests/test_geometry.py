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


def test_quat_vector_mul():
    from alphafold.utils.geometry import conjugate_quat, quat_vector_mul

    q = torch.tensor([0.3, -0.4, 0.1, 0.8])
    q = q / torch.linalg.vector_norm(q)
    q_copy = q.clone()
    v = torch.tensor([4.0, 1.0, 2.0])

    q_conj = conjugate_quat(q)
    q_conj_batch = conjugate_quat(q.broadcast_to((3, 4, 4)))

    q_conj_exp = torch.load(f'{control_folder}/quat_conjugate_check.pt')
    q_conj_batch_exp = q_conj_exp.broadcast_to(3, 4, 4)

    assert torch.allclose(
        q_copy, q, atol=1e-5
    ), 'Conjugation is performed in-place (modifies q) but should be done out-of-place. Clone the quaternion before modification.'
    assert torch.allclose(q_conj, q_conj_exp,
                          atol=1e-5), 'Error in q_conj, single use.'
    assert torch.allclose(q_conj_batch, q_conj_batch_exp,
                          atol=1e-5), 'Error in q_conj, batched use.'

    qv = quat_vector_mul(q, v)
    qv_batch = quat_vector_mul(q.broadcast_to(3, 4, 4),
                               v.broadcast_to(3, 4, 3))

    qv_exp = torch.load(f'{control_folder}/quat_vector_check.pt')
    qv_batch_exp = qv_exp.broadcast_to(3, 4, 3)

    assert torch.allclose(qv, qv_exp,
                          atol=1e-5), 'Error in quat_vector_mul, single use.'
    assert torch.allclose(qv_batch, qv_batch_exp,
                          atol=1e-5), 'Error in quat_vector_mul, batched use.'


def test_quat_to_3x3_rotation():
    from alphafold.utils.geometry import quat_to_3x3_rotation

    quat = torch.tensor([0.9830, 0.1294, 0.1294, 0.0170])

    R = quat_to_3x3_rotation(quat)

    a, b, c, d = torch.unbind(quat, dim=-1)

    exp_R = [[
        a**2 + b**2 - c**2 - d**2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c
    ], [
        2 * b * c + 2 * a * d, a**2 - b**2 + c**2 - d**2, 2 * c * d - 2 * a * b
    ], [
        2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a**2 - b**2 - c**2 + d**2
    ]]
    exp_R = [torch.stack(vals, dim=-1) for vals in exp_R]
    exp_R = torch.stack(exp_R, dim=-2)

    assert torch.allclose(R, exp_R, atol=1e-5)
