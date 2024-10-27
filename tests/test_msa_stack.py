control_folder = "./control_values/evoformer"


def test_MSARowAttentionWithPairBias():
    from alphafold.model.msa_stack import MSARowAttentionWithPairBias
    from tests.control_values.evoformer.evoformer_checks import c_m, c_z, c, N_head
    from tests.control_values.evoformer.evoformer_checks import test_module_shape, test_module

    msa_row_att = MSARowAttentionWithPairBias(c_m, c_z, c, N_head)

    test_module_shape(msa_row_att, 'msa_row_att', control_folder)

    test_module(msa_row_att, 'msa_row_att', ('m', 'z'), 'out', control_folder)


def test_MSAColumnAttention():
    from alphafold.model.msa_stack import MSAColumnAttention
    from tests.control_values.evoformer.evoformer_checks import c_m, c, N_head
    from tests.control_values.evoformer.evoformer_checks import test_module_shape, test_module

    msa_col_att = MSAColumnAttention(c_m, c, N_head)

    test_module_shape(msa_col_att, 'msa_col_att', control_folder)

    test_module(msa_col_att, 'msa_col_att', 'm', 'out', control_folder)


def test_MSATransition():
    from alphafold.model.msa_stack import MSATransition
    from tests.control_values.evoformer.evoformer_checks import c_m
    from tests.control_values.evoformer.evoformer_checks import test_module_shape, test_module

    n = 3
    msa_trans = MSATransition(c_m, n)

    test_module_shape(msa_trans, 'msa_transition', control_folder)

    test_module(msa_trans, 'msa_transition', 'm', 'out', control_folder)
