"""
unittests for table outputter
"""
import pytest
import salt.output.table_out as table_out
import salt.utils.stringutils

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {table_out: {}}

@pytest.fixture
def data():
    if False:
        return 10
    return [{'Food': salt.utils.stringutils.to_str('яйца, бекон, колбаса и спам'), 'Price': 5.99}, {'Food': 'спам, спам, спам, яйца и спам', 'Price': 3.99}]

def test_output(data):
    if False:
        while True:
            i = 10
    ret = table_out.output(data)
    assert ret == '    -----------------------------------------\n    |              Food             | Price |\n    -----------------------------------------\n    |  яйца, бекон, колбаса и спам  |  5.99 |\n    -----------------------------------------\n    | спам, спам, спам, яйца и спам |  3.99 |\n    -----------------------------------------'