from hypothesis import strategies as st
from tests.common.debug import find_any

def test_can_generate_large_lists_with_min_size():
    if False:
        return 10
    find_any(st.lists(st.integers(), min_size=400))