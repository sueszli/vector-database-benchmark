import threading
from hypothesis import given, strategies as st

def test_can_run_given_in_thread():
    if False:
        i = 10
        return i + 15
    has_run_successfully = [False]

    @given(st.integers())
    def test(n):
        if False:
            print('Hello World!')
        has_run_successfully[0] = True
    t = threading.Thread(target=test)
    t.start()
    t.join()
    assert has_run_successfully[0]