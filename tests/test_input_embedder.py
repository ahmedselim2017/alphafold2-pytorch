control_folder = "./control_values/feature_embedder"


def test_InputEmbedder_init():
    from alphafold.model.input_embedder import InputEmbedder

    from tests.control_values.feature_embedder.embedding_checks import (
        test_module_forward, test_module_method, c_m, c_z, tf_dim,
        msa_feat_dim)

    input_embedder = InputEmbedder(c_m,
                                   c_z,
                                   tf_dim,
                                   msa_feat_dim=msa_feat_dim,
                                   vbins=32)

    test_module_method(input_embedder, 'input_embedder_relpos',
                       'residue_index', 'z_out', control_folder,
                       lambda x: input_embedder.relpos(x))

    input_embedder = InputEmbedder(c_m,
                                   c_z,
                                   tf_dim,
                                   msa_feat_dim=msa_feat_dim,
                                   vbins=32)

    test_module_forward(input_embedder, 'input_embedder', 'batch',
                        ('m_out', 'z_out'), control_folder)
