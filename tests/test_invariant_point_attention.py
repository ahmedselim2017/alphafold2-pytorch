control_folder = "./control_values/structure_module"


def test_prepare_qkv():
    from alphafold.model.invariant_point_attention import InvariantPointAttention
    from tests.control_values.structure_module.structure_module_checks import test_module_shape, test_module_method
    from tests.control_values.structure_module.structure_module_checks import c_s, c_z, n_qp, n_pv, N_head, c

    ipa = InvariantPointAttention(c_s, c_z, n_qp, n_pv, N_head, c)

    test_module_shape(ipa, 'ipa', control_folder)

    test_module_method(ipa, 'ipa_prep', 's', ('q', 'k', 'v', 'qp', 'kp', 'vp'),
                       control_folder, lambda x: ipa.prepare_qkv(x))
