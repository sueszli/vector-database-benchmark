import configparser
import os
import pytest

@pytest.mark.parametrize('um3_file, um3e_file', [('ultimaker3_aa0.25.inst.cfg', 'ultimaker3_extended_aa0.25.inst.cfg'), ('ultimaker3_aa0.8.inst.cfg', 'ultimaker3_extended_aa0.8.inst.cfg'), ('ultimaker3_aa04.inst.cfg', 'ultimaker3_extended_aa04.inst.cfg'), ('ultimaker3_bb0.8.inst.cfg', 'ultimaker3_extended_bb0.8.inst.cfg'), ('ultimaker3_bb04.inst.cfg', 'ultimaker3_extended_bb04.inst.cfg')])
def test_ultimaker3extended_variants(um3_file, um3e_file):
    if False:
        for i in range(10):
            print('nop')
    directory = os.path.join(os.path.dirname(__file__), '..', 'resources', 'variants')
    um3 = configparser.ConfigParser()
    um3.read_file(open(os.path.join(directory, um3_file), encoding='utf-8'))
    um3e = configparser.ConfigParser()
    um3e.read_file(open(os.path.join(directory, um3e_file), encoding='utf-8'))
    assert [value for value in um3['values']] == [value for value in um3e['values']]