import pytest
from hypothesis import given, strategies as st
from tests.common.debug import minimal

@given(st.lists(st.uuids()))
def test_are_unique(ls):
    if False:
        return 10
    assert len(set(ls)) == len(ls)

def test_retains_uniqueness_in_simplify():
    if False:
        for i in range(10):
            print('nop')
    ts = minimal(st.lists(st.uuids()), lambda x: len(x) >= 5)
    assert len(ts) == len(set(ts)) == 5

@pytest.mark.parametrize('version', (1, 2, 3, 4, 5))
def test_can_generate_specified_version(version):
    if False:
        for i in range(10):
            print('nop')

    @given(st.uuids(version=version))
    def inner(uuid):
        if False:
            while True:
                i = 10
        assert version == uuid.version
    inner()