control_folder = "./control_values/structure_module"


def test_StructureModuleTransition():
    from alphafold.model.structure_module import StructureModuleTransition
    from tests.control_values.structure_module.structure_module_checks import test_module_forward, test_module_shape, c_s

    transition = StructureModuleTransition(c_s)

    test_module_shape(transition, 'sm_transition', control_folder)

    test_module_forward(transition, 'sm_transition', 's', 's_out',
                        control_folder)
