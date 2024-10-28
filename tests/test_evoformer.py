control_folder = "./control_values/evoformer"


def test_EvoformerBlock():
    from alphafold.model.evoformer import EvoformerBlock
    from tests.control_values.evoformer.evoformer_checks import c_m, c_z
    from tests.control_values.evoformer.evoformer_checks import test_module_shape, test_module

    evo_block = EvoformerBlock(c_m, c_z)

    test_module_shape(evo_block, 'evo_block', control_folder)

    test_module(evo_block, 'evo_block', ('m', 'z'), ('m_out', 'z_out'),
                control_folder)
