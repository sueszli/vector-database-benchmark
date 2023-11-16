import os
from pathlib import Path
from typing import Optional, Tuple
from venv import EnvBuilder
import pytest
from pip._internal.cli.cmdoptions import _convert_python_version
from pip._internal.cli.main_parser import identify_python_interpreter

@pytest.mark.parametrize('value, expected', [('', (None, None)), ('2', ((2,), None)), ('3', ((3,), None)), ('3.7', ((3, 7), None)), ('3.7.3', ((3, 7, 3), None)), ('34', ((3, 4), None)), ('310', ((3, 10), None)), ('ab', ((), 'each version part must be an integer')), ('3a', ((), 'each version part must be an integer')), ('3.7.a', ((), 'each version part must be an integer')), ('3.7.3.1', ((), 'at most three version parts are allowed'))])
def test_convert_python_version(value: str, expected: Tuple[Optional[Tuple[int, ...]], Optional[str]]) -> None:
    if False:
        i = 10
        return i + 15
    actual = _convert_python_version(value)
    assert actual == expected, f'actual: {actual!r}'

def test_identify_python_interpreter_venv(tmpdir: Path) -> None:
    if False:
        i = 10
        return i + 15
    env_path = tmpdir / 'venv'
    env = EnvBuilder(with_pip=False)
    env.create(env_path)
    interp = identify_python_interpreter(os.fspath(env_path))
    assert interp is not None
    assert Path(interp).exists()
    assert identify_python_interpreter(interp) == interp
    assert identify_python_interpreter(str(tmpdir / 'nonexistent')) is None