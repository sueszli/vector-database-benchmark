from hypothesis import given, strategies as st

@given(st.data())
def test_never_draw_anything(data):
    if False:
        while True:
            i = 10
    pass