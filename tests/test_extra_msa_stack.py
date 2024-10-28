control_folder = "./control_values/feature_embedder"


def test_ExtraMSAEmbedder():
    from alphafold.model.extra_msa_stack import ExtraMSAEmbedder
    from tests.control_values.feature_embedder.embedding_checks import test_module_shape, test_module_forward, f_e, c_e

    extra_msa_embedder = ExtraMSAEmbedder(f_e, c_e)

    test_module_shape(extra_msa_embedder, 'extra_msa_embedder', control_folder)

    test_module_forward(extra_msa_embedder, 'extra_msa_embedder', 'batch',
                        'e_out', control_folder)


def test_MSAColumnGlobalAttention():
    from alphafold.model.extra_msa_stack import MSAColumnGlobalAttention
    from tests.control_values.feature_embedder.embedding_checks import (
        test_module_shape, test_module_forward, c_m, c, N_head)

    msa_global_col_att = MSAColumnGlobalAttention(c_m, c, N_head)

    test_module_shape(msa_global_col_att, 'msa_global_col_att', control_folder)

    test_module_forward(msa_global_col_att, 'msa_global_col_att', 'm', 'm_out',
                        control_folder)


def test_ExtraMSABlock():
    from alphafold.model.extra_msa_stack import ExtraMSABlock
    from tests.control_values.feature_embedder.embedding_checks import test_module_shape, test_module_forward, c_e, c_z

    extra_msa_block = ExtraMSABlock(c_e, c_z)

    test_module_shape(extra_msa_block, 'extra_msa_block', control_folder)

    test_module_forward(extra_msa_block, 'extra_msa_block', ('e', 'z'),
                        'm_out', control_folder)


def test_ExtraMSAStack():
    from alphafold.model.extra_msa_stack import ExtraMSAStack
    from tests.control_values.feature_embedder.embedding_checks import test_module_shape, test_module_forward, c_e, c_z

    num_blocks = 3
    extra_msa_stack = ExtraMSAStack(c_e, c_z, num_blocks)

    test_module_shape(extra_msa_stack, 'extra_msa_stack', control_folder)

    test_module_forward(extra_msa_stack, 'extra_msa_stack', ('e', 'z'),
                        'm_out', control_folder)
