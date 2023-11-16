import tempfile
import pytest

@pytest.fixture(scope='module')
def minion_config_overrides():
    if False:
        for i in range(10):
            print('nop')
    with tempfile.TemporaryDirectory() as tempdir:
        yield {'mydude': {'driver': 'sqlite3', 'database': tempdir + '/test_sdb.sq3', 'table': __name__, 'create_table': True}}

@pytest.mark.parametrize('expected_value', ('foo', b'bang', ['cool', b'guy', 'dude', b'\x001\x99B'], {'this': b'has some', b'complicated': 'things', 'all': [{'going': 'on'}, {'but': 'that', 42: 'should be fine'}]}))
def test_setting_sdb_values_with_text_and_bytes_should_retain_data_types(expected_value, modules):
    if False:
        while True:
            i = 10
    modules.sdb.set('sdb://mydude/fnord', expected_value)
    actual_value = modules.sdb.get('sdb://mydude/fnord', strict=True)
    assert actual_value == expected_value