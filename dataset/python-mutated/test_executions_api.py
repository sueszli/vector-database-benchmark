import pytest
from modin.core.storage_formats import BaseQueryCompiler, PandasQueryCompiler
from modin.experimental.core.storage_formats.pyarrow import PyarrowQueryCompiler
BASE_EXECUTION = BaseQueryCompiler
EXECUTIONS = [PandasQueryCompiler, PyarrowQueryCompiler]

def test_base_abstract_methods():
    if False:
        for i in range(10):
            print('nop')
    allowed_abstract_methods = ['__init__', 'free', 'finalize', 'execute', 'to_pandas', 'from_pandas', 'from_arrow', 'default_to_pandas', 'from_dataframe', 'to_dataframe']
    not_implemented_methods = BASE_EXECUTION.__abstractmethods__.difference(allowed_abstract_methods)
    not_implemented_methods = list(not_implemented_methods)
    not_implemented_methods.sort()
    assert len(not_implemented_methods) == 0, f'{BASE_EXECUTION} has not implemented abstract methods: {not_implemented_methods}'

@pytest.mark.parametrize('execution', EXECUTIONS)
def test_api_consistent(execution):
    if False:
        for i in range(10):
            print('nop')
    base_methods = set(BASE_EXECUTION.__dict__)
    custom_methods = set([key for key in execution.__dict__.keys() if not key.startswith('_')])
    extra_methods = custom_methods.difference(base_methods)
    assert len(extra_methods) == 0, f'{execution} implement these extra methods: {extra_methods}'