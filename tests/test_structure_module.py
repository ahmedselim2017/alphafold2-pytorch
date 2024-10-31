control_folder = "./control_values/structure_module"


def test_StructureModuleTransition():
    from alphafold.model.structure_module import StructureModuleTransition
    from tests.control_values.structure_module.structure_module_checks import test_module_forward, test_module_shape, c_s

    transition = StructureModuleTransition(c_s)

    test_module_shape(transition, 'sm_transition', control_folder)

    test_module_forward(transition, 'sm_transition', 's', 's_out',
                        control_folder)


def test_BackboneUpdate():
    from alphafold.model.structure_module import BackboneUpdate
    from tests.control_values.structure_module.structure_module_checks import test_module_forward, test_module_shape, c_s

    bb_update = BackboneUpdate(c_s)

    test_module_shape(bb_update, 'bb_update', control_folder)

    test_module_forward(bb_update, 'bb_update', 's', 'T_out', control_folder)


def test_AngleResNetLayer():
    from alphafold.model.structure_module import AngleResNetLayer

    from tests.control_values.structure_module.structure_module_checks import test_module_forward, test_module_shape, c

    resnet_layer = AngleResNetLayer(c)

    test_module_shape(resnet_layer, 'resnet_layer', control_folder)

    test_module_forward(resnet_layer, 'resnet_layer', 'a', 'a_out',
                        control_folder)


def test_AngleResNet():
    from alphafold.model.structure_module import AngleResNet

    from tests.control_values.structure_module.structure_module_checks import test_module_forward, test_module_shape, c_s, c

    angle_resnet = AngleResNet(c_s, c)

    test_module_shape(angle_resnet, 'angle_resnet', control_folder)

    test_module_forward(angle_resnet, 'angle_resnet', ('s', 's_initial'),
                        'alpha', control_folder)


def test_StructureModule_init():
    from tests.control_values.structure_module.structure_module_checks import test_module_shape, c_s, c_z, c, n_layer
    from alphafold.model.structure_module import StructureModule

    sm = StructureModule(c_s, c_z, n_layer, c)

    test_module_shape(sm, 'structure_module', control_folder)


def test_StructureModule_process_outputs():
    from tests.control_values.structure_module.structure_module_checks import test_module_method, c_s, c_z, c, n_layer
    from alphafold.model.structure_module import StructureModule

    sm = StructureModule(c_s, c_z, n_layer, c)

    test_module_method(sm,
                       'sm_process_outputs', ('T', 'alpha', 'F'),
                       ('pos', 'pos_mask', 'pseudo_beta'),
                       control_folder,
                       lambda *x: sm.process_outputs(*x),
                       include_batched=False)


def test_StructureModule():
    from tests.control_values.structure_module.structure_module_checks import test_module_method, c_s, c_z, c, n_layer
    from alphafold.model.structure_module import StructureModule

    sm = StructureModule(c_s, c_z, n_layer, c)

    def check(*args):
        output = sm(*args)
        return output['angles'], output['frames'], output[
            'final_positions'], output['position_mask'], output[
                'pseudo_beta_positions']

    test_module_method(sm, 'structure_module', ('s', 'z', 'F'),
                       ('angles', 'frames', 'final_positions', 'position_mask',
                        'pseudo_beta_positions'), control_folder, check)
