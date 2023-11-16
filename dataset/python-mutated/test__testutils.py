import sys
from scipy._lib._testutils import _parse_size, _get_mem_available
import pytest

def test__parse_size():
    if False:
        return 10
    expected = {'12': 12000000.0, '12 b': 12, '12k': 12000.0, '  12  M  ': 12000000.0, '  12  G  ': 12000000000.0, ' 12Tb ': 12000000000000.0, '12  Mib ': 12 * 1024.0 ** 2, '12Tib': 12 * 1024.0 ** 4}
    for (inp, outp) in sorted(expected.items()):
        if outp is None:
            with pytest.raises(ValueError):
                _parse_size(inp)
        else:
            assert _parse_size(inp) == outp

def test__mem_available():
    if False:
        while True:
            i = 10
    available = _get_mem_available()
    if sys.platform.startswith('linux'):
        assert available >= 0
    else:
        assert available is None or available >= 0