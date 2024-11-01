control_folder = "./control_values/model"


def test_model():

    from alphafold.model.model import Model
    from tests.control_values.model.model_checks import c_m, c_z, c_e, f_e, tf_dim, c_s, num_blocks_evoformer, num_blocks_extra_msa
    from tests.control_values.model.model_checks import test_module_shape, test_module_method

    model = Model(c_m, c_z, c_e, f_e, tf_dim, c_s, num_blocks_extra_msa,
                  num_blocks_evoformer)

    test_module_shape(model, 'model', control_folder)

    def test_method(*args):
        outputs = model(*args)
        return outputs['final_positions'], outputs['position_mask'], outputs[
            'angles'], outputs['frames']

    test_module_method(
        model, 'model', 'batch',
        ('final_positions', 'position_mask', 'angles', 'frames'),
        control_folder, test_method)
