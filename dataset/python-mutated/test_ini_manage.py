import os
import salt.modules.ini_manage

def test_section_req():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the __repr__ in the _Section class\n    '
    expected = '_Section(){}{{}}'.format(os.linesep)
    assert repr(salt.modules.ini_manage._Section('test')) == expected