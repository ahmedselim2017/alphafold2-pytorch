control_folder = "./control_values/evoformer"


def test_trianglemultiplication():
    from alphafold.model.pair_stack import TriangleMultiplication

    from tests.control_values.evoformer.evoformer_checks import c_z, c
    from tests.control_values.evoformer.evoformer_checks import test_module_shape, test_module

    tri_mul_in = TriangleMultiplication(c_z, 'incoming', c)
    tri_mul_out = TriangleMultiplication(c_z, 'outgoing', c)

    test_module_shape(tri_mul_in, 'tri_mul_in', control_folder)
    test_module_shape(tri_mul_out, 'tri_mul_out', control_folder)

    test_module(tri_mul_in, 'tri_mul_in', 'z', 'z_out', control_folder)
    test_module(tri_mul_out, 'tri_mul_out', 'z', 'z_out', control_folder)


def test_TriangleAttention():
    from alphafold.model.pair_stack import TriangleAttention
    from tests.control_values.evoformer.evoformer_checks import c_z, c, N_head
    from tests.control_values.evoformer.evoformer_checks import test_module_shape, test_module

    tri_att_start = TriangleAttention(c_z, 'starting_node', c, N_head)
    tri_att_end = TriangleAttention(c_z, 'ending_node', c, N_head)

    test_module_shape(tri_att_start, 'tri_att_start', control_folder)
    test_module_shape(tri_att_end, 'tri_att_end', control_folder)

    test_module(tri_att_start, 'tri_att_start', 'z', 'z_out', control_folder)
    test_module(tri_att_end, 'tri_att_end', 'z', 'z_out', control_folder)


def test_PairTransition():
    from alphafold.model.pair_stack import PairTransition
    from tests.control_values.evoformer.evoformer_checks import c_z
    from tests.control_values.evoformer.evoformer_checks import test_module_shape, test_module

    n = 3
    pair_trans = PairTransition(c_z, n)

    test_module_shape(pair_trans, 'pair_transition', control_folder)

    test_module(pair_trans, 'pair_transition', 'z', 'z_out', control_folder)
