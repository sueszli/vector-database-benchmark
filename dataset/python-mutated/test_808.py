import lief
from utils import get_sample

def test_core_offset_0():
    if False:
        for i in range(10):
            print('nop')
    file = get_sample('ELF/ELF_Core_issue_808.core')
    core = lief.parse(file)
    assert len(core.notes) == 7