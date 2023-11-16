from vyper.codegen import core
from vyper.compiler.phases import CompilerData
from vyper.compiler.settings import OptimizationLevel, _is_debug_mode

def test_default_settings():
    if False:
        for i in range(10):
            print('nop')
    source_code = ''
    compiler_data = CompilerData(source_code)
    _ = compiler_data.vyper_module
    assert compiler_data.settings.optimize == OptimizationLevel.GAS

def test_default_opt_level():
    if False:
        for i in range(10):
            print('nop')
    assert OptimizationLevel.default() == OptimizationLevel.GAS

def test_codegen_opt_level():
    if False:
        for i in range(10):
            print('nop')
    assert core._opt_level == OptimizationLevel.GAS
    assert core._opt_gas() is True
    assert core._opt_none() is False
    assert core._opt_codesize() is False

def test_debug_mode(pytestconfig):
    if False:
        return 10
    debug_mode = pytestconfig.getoption('enable_compiler_debug_mode')
    assert _is_debug_mode() == debug_mode