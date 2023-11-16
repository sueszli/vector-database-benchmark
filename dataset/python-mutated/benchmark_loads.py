from json import loads as json_loads
import pytest
from .data import fixtures, libraries
from .util import read_fixture

@pytest.mark.parametrize('fixture', fixtures)
@pytest.mark.parametrize('library', libraries)
def test_loads(benchmark, fixture, library):
    if False:
        i = 10
        return i + 15
    (dumper, loader) = libraries[library]
    benchmark.group = f'{fixture} deserialization'
    benchmark.extra_info['lib'] = library
    data = read_fixture(f'{fixture}.xz')
    benchmark.extra_info['correct'] = json_loads(dumper(loader(data))) == json_loads(data)
    benchmark(loader, data)