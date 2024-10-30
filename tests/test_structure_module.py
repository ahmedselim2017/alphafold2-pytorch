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


def test_aAngleResNetLayer():
    from alphafold.model.structure_module import AngleResNetLayer

    from tests.control_values.structure_module.structure_module_checks import test_module_forward, test_module_shape, c

    resnet_layer = AngleResNetLayer(c)

    test_module_shape(resnet_layer, 'resnet_layer', control_folder)

    test_module_forward(resnet_layer, 'resnet_layer', 'a', 'a_out',
                        control_folder)
