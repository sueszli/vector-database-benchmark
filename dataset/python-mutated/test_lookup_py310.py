from hypothesis import strategies as st
from tests.common.debug import find_any

def test_native_unions():
    if False:
        print('Hello World!')
    s = st.from_type(int | list[str])
    find_any(s, lambda x: isinstance(x, int))
    find_any(s, lambda x: isinstance(x, list))