control_folder = "./control_values"


def test_init():
    from tests.control_values.attention_checks import c_in, c, N_head, attn_dim
    from tests.control_values.attention_checks import test_module_shape

    from alphafold.model.multi_head_attention import MultiHeadAttention

    mha = MultiHeadAttention(c_in, c, N_head, attn_dim, gated=True)
    mha_bias = MultiHeadAttention(c_in,
                                  c,
                                  N_head,
                                  attn_dim,
                                  gated=True,
                                  use_bias_for_embeddings=True)

    test_module_shape(mha, 'mha_init', control_folder)
    test_module_shape(mha_bias, 'mha_bias_init', control_folder)


def test_prepare_qkv():
    from alphafold.model.multi_head_attention import MultiHeadAttention
    from tests.control_values.attention_checks import c_in, c, N_head, attn_dim
    from tests.control_values.attention_checks import test_module_method

    mha = MultiHeadAttention(c_in, c, N_head, attn_dim=attn_dim, gated=True)

    test_module_method(mha, 'mha_prep_qkv', ('q', 'k', 'v'),
                       ('q_prep', 'k_prep', 'v_prep'), control_folder,
                       mha.prepare_qkv)


def test_forward():
    from alphafold.model.multi_head_attention import MultiHeadAttention
    from tests.control_values.attention_checks import c_in, c, N_head, attn_dim
    from tests.control_values.attention_checks import test_module_forward

    mha_ungated = MultiHeadAttention(c_in,
                                     c,
                                     N_head,
                                     attn_dim=attn_dim,
                                     gated=False)
    test_module_forward(mha_ungated, 'mha_ungated_forward', 'x', 'out',
                        control_folder)

    mha_gated = MultiHeadAttention(c_in,
                                   c,
                                   N_head,
                                   attn_dim=attn_dim,
                                   gated=True)
    test_module_forward(mha_ungated, 'mha_gated_forward', 'x', 'out',
                        control_folder)

    test_module_forward(mha_ungated, 'mha_gated_bias_forward', ('x', 'bias'),
                        'out', control_folder)


def test_forward_global():
    from alphafold.model.multi_head_attention import MultiHeadAttention
    from tests.control_values.attention_checks import c_in, c, N_head, attn_dim
    from tests.control_values.attention_checks import test_module_forward
    from tests.control_values.attention_checks import test_module_shape
    from tests.control_values.attention_checks import test_module_method

    mha_global = MultiHeadAttention(c_in,
                                    c,
                                    N_head,
                                    attn_dim,
                                    gated=False,
                                    is_global=True)

    test_module_shape(mha_global, 'mha_global_init', control_folder)

    mha_global = MultiHeadAttention(c_in,
                                    c,
                                    N_head,
                                    attn_dim,
                                    gated=False,
                                    is_global=True)

    test_module_method(mha_global, 'mha_global_prep_qkv',
                       ('q_global', 'k_global', 'v_global'), ('q', 'k', 'v'),
                       control_folder, mha_global.prepare_qkv_global)

    mha_global = MultiHeadAttention(c_in,
                                    c,
                                    N_head,
                                    attn_dim,
                                    gated=False,
                                    is_global=True)

    test_module_forward(mha_global, 'mha_global_forward', 'x', 'out',
                        control_folder)
