from panda3d import core

def test_shaderattrib_flags():
    if False:
        return 10
    shattr = core.ShaderAttrib.make()
    shattr = shattr.set_flag(core.ShaderAttrib.F_hardware_skinning, True)
    assert shattr.get_flag(core.ShaderAttrib.F_hardware_skinning)
    assert not shattr.get_flag(core.ShaderAttrib.F_subsume_alpha_test)
    shattr = shattr.clear_flag(core.ShaderAttrib.F_hardware_skinning)
    assert not shattr.get_flag(core.ShaderAttrib.F_hardware_skinning)
    shattr = shattr.set_flag(core.ShaderAttrib.F_hardware_skinning, False)
    assert not shattr.get_flag(core.ShaderAttrib.F_hardware_skinning)
    shattr = core.ShaderAttrib.make()
    shattr = shattr.set_flag(core.ShaderAttrib.F_hardware_skinning | core.ShaderAttrib.F_subsume_alpha_test, True)
    assert shattr.get_flag(core.ShaderAttrib.F_hardware_skinning | core.ShaderAttrib.F_subsume_alpha_test)
    assert not shattr.get_flag(core.ShaderAttrib.F_shader_point_size)
    assert not shattr.get_flag(core.ShaderAttrib.F_disable_alpha_write | core.ShaderAttrib.F_shader_point_size)
    shattr = shattr.clear_flag(core.ShaderAttrib.F_hardware_skinning | core.ShaderAttrib.F_subsume_alpha_test)
    assert not shattr.get_flag(core.ShaderAttrib.F_hardware_skinning | core.ShaderAttrib.F_subsume_alpha_test)
    shattr = shattr.set_flag(core.ShaderAttrib.F_hardware_skinning | core.ShaderAttrib.F_subsume_alpha_test, False)
    assert not shattr.get_flag(core.ShaderAttrib.F_hardware_skinning | core.ShaderAttrib.F_subsume_alpha_test)

def test_shaderattrib_compare():
    if False:
        while True:
            i = 10
    shattr1 = core.ShaderAttrib.make()
    shattr2 = core.ShaderAttrib.make()
    assert shattr1.compare_to(shattr2) == 0
    assert shattr2.compare_to(shattr1) == 0
    shattr2 = core.ShaderAttrib.make().set_flag(core.ShaderAttrib.F_subsume_alpha_test, False)
    assert shattr1.compare_to(shattr2) != 0
    assert shattr2.compare_to(shattr1) != 0
    shattr1 = core.ShaderAttrib.make().set_flag(core.ShaderAttrib.F_subsume_alpha_test, False)
    assert shattr1.compare_to(shattr2) == 0
    assert shattr2.compare_to(shattr1) == 0
    shattr2 = core.ShaderAttrib.make().set_flag(core.ShaderAttrib.F_subsume_alpha_test, True)
    assert shattr1.compare_to(shattr2) != 0
    assert shattr2.compare_to(shattr1) != 0