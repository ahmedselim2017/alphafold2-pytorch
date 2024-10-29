import torch


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
