import os
import re
import subprocess
import tempfile
from typing import List
import pytest

def get_pyright_reveal_type_output(filename) -> List[str]:
    if False:
        return 10
    stdout = subprocess.check_output(['pyright', filename]).decode('utf-8')
    match = re.findall('Type of "(?:[^"]+)" is "([^"]+)"', stdout)
    assert match
    return match

def get_mypy_type_output(filename) -> List[str]:
    if False:
        i = 10
        return i + 15
    stdout = subprocess.check_output(['mypy', filename]).decode('utf-8')
    match = re.findall('note: Revealed type is "([^"]+)"', stdout)
    assert match
    return match

@pytest.mark.typesignature
def test_type_signatures_constructor_nested_resource():
    if False:
        while True:
            i = 10
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, 'test.py')
        with open(filename, 'w') as f:
            f.write('\nfrom dagster import ConfigurableResource\n\nclass InnerResource(ConfigurableResource):\n    a_string: str\n\nclass OuterResource(ConfigurableResource):\n    inner: InnerResource\n    a_bool: bool\n\nreveal_type(InnerResource.__init__)\nreveal_type(OuterResource.__init__)\n\nmy_outer = OuterResource(inner=InnerResource(a_string="foo"), a_bool=True)\nreveal_type(my_outer.inner)\n')
        pyright_out = get_pyright_reveal_type_output(filename)
        mypy_out = get_mypy_type_output(filename)
        assert pyright_out[0] == '(self: InnerResource, *, a_string: str) -> None'
        assert pyright_out[1] == '(self: OuterResource, *, inner: InnerResource | PartialResource[InnerResource], a_bool: bool) -> None'
        assert pyright_out[2] == 'InnerResource'
        assert mypy_out[2] == 'test.InnerResource'

@pytest.mark.typesignature
def test_type_signatures_config_at_launch():
    if False:
        for i in range(10):
            print('nop')
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, 'test.py')
        with open(filename, 'w') as f:
            f.write('\nfrom dagster import ConfigurableResource\n\nclass MyResource(ConfigurableResource):\n    a_string: str\n\nreveal_type(MyResource.configure_at_launch())\n')
        pyright_out = get_pyright_reveal_type_output(filename)
        mypy_out = get_mypy_type_output(filename)
        assert pyright_out[0] == 'PartialResource[MyResource]'
        assert mypy_out[0].endswith('PartialResource[test.MyResource]')

@pytest.mark.typesignature
def test_type_signatures_constructor_resource_dependency():
    if False:
        i = 10
        return i + 15
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, 'test.py')
        with open(filename, 'w') as f:
            f.write('\nfrom dagster import ConfigurableResource, ResourceDependency\n\nclass StringDependentResource(ConfigurableResource):\n    a_string: ResourceDependency[str]\n\nreveal_type(StringDependentResource.__init__)\n\nmy_str_resource = StringDependentResource(a_string="foo")\nreveal_type(my_str_resource.a_string)\n')
        pyright_out = get_pyright_reveal_type_output(filename)
        mypy_out = get_mypy_type_output(filename)
        assert pyright_out[0] == '(self: StringDependentResource, *, a_string: ConfigurableResourceFactory[str] | PartialResource[str] | ResourceDefinition | str) -> None'
        assert pyright_out[1] == 'str'
        assert mypy_out[1] == 'builtins.str'

@pytest.mark.typesignature
def test_type_signatures_alias():
    if False:
        print('Hello World!')
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, 'test.py')
        with open(filename, 'w') as f:
            f.write('\nfrom dagster import ConfigurableResource\nfrom pydantic import Field\n\nclass ResourceWithAlias(ConfigurableResource):\n    _schema: str = Field(alias="schema")\n\nreveal_type(ResourceWithAlias.__init__)\n\nmy_resource = ResourceWithAlias(schema="foo")\n')
        pyright_out = get_pyright_reveal_type_output(filename)
        assert pyright_out[0] == '(self: ResourceWithAlias, *, schema: str) -> None'