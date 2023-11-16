import pytest
from vyper.compiler.phases import CompilerData
from vyper.compiler.settings import OptimizationLevel, Settings
codes = ['\ns: uint256\n\n@internal\ndef ctor_only():\n    self.s = 1\n\n@internal\ndef runtime_only():\n    self.s = 2\n\n@external\ndef bar():\n    self.runtime_only()\n\n@external\ndef __init__():\n    self.ctor_only()\n    ', '\ns: uint256\n\n@internal\ndef runtime_only():\n    self.s = 1\n\n@internal\ndef foo():\n    self.runtime_only()\n\n@internal\ndef ctor_only():\n    self.s += 1\n\n@external\ndef bar():\n    self.foo()\n\n@external\ndef __init__():\n    self.ctor_only()\n    ', '\ns: uint256\n\n@internal\ndef ctor_only():\n    self.s = 1\n\n@internal\ndef runtime_only():\n    for i in range(10):\n        self.s += 1\n\n@external\ndef bar():\n    self.runtime_only()\n\n@external\ndef __init__():\n    self.ctor_only()\n    ']

@pytest.mark.parametrize('code', codes)
def test_dead_code_eliminator(code):
    if False:
        while True:
            i = 10
    c = CompilerData(code, settings=Settings(optimize=OptimizationLevel.NONE))
    initcode_asm = [i for i in c.assembly if not isinstance(i, list)]
    runtime_asm = c.assembly_runtime
    ctor_only_label = '_sym_internal_ctor_only___'
    runtime_only_label = '_sym_internal_runtime_only___'
    assert ctor_only_label + '_deploy' in initcode_asm
    assert runtime_only_label + '_deploy' not in initcode_asm
    for s in (ctor_only_label, runtime_only_label):
        assert s + '_runtime' in runtime_asm
    c = CompilerData(code, settings=Settings(optimize=OptimizationLevel.GAS))
    initcode_asm = [i for i in c.assembly if not isinstance(i, list)]
    runtime_asm = c.assembly_runtime
    for instr in runtime_asm:
        if isinstance(instr, str):
            assert not instr.startswith(ctor_only_label), instr
    for instr in initcode_asm:
        if isinstance(instr, str):
            assert not instr.startswith(runtime_only_label), instr