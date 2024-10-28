control_folder = "./control_values/feature_embedder"


def test_recycling_embedder():

    from alphafold.model.recycling_embedder import RecyclingEmbedder

    from tests.control_values.feature_embedder.embedding_checks import (
        c_m, c_z, test_module_shape, test_module_forward)

    recycling_embedder = RecyclingEmbedder(c_m, c_z)

    test_module_shape(recycling_embedder, 'recycling_embedder', control_folder)

    test_module_forward(recycling_embedder, 'recycling_embedder',
                        ('m', 'z', 'x'), ('m_out', 'z_out'), control_folder)
