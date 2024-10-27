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
