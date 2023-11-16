import os
import platform
import pytest
from panda3d import core
SHADERS_DIR = core.Filename.from_os_specific(os.path.dirname(__file__))

def run_cg_compile_check(gsg, shader_path, expect_fail=False):
    if False:
        for i in range(10):
            print('nop')
    'Compile supplied Cg shader path and check for errors'
    shader = core.Shader.load(shader_path, core.Shader.SL_Cg)
    if expect_fail:
        assert shader is None
    else:
        assert shader is not None

@pytest.mark.skipif(platform.machine().lower() == 'arm64', reason='Cg not supported on arm64')
def test_cg_compile_error(gsg):
    if False:
        return 10
    'Test getting compile errors from bad Cg shaders'
    shader_path = core.Filename(SHADERS_DIR, 'cg_bad.sha')
    run_cg_compile_check(gsg, shader_path, expect_fail=True)

@pytest.mark.skipif(platform.machine().lower() == 'arm64', reason='Cg not supported on arm64')
def test_cg_from_file(gsg):
    if False:
        return 10
    'Test compiling Cg shaders from files'
    shader_path = core.Filename(SHADERS_DIR, 'cg_simple.sha')
    run_cg_compile_check(gsg, shader_path)