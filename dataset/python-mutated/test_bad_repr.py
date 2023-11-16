from hypothesis import given, strategies as st

class BadRepr:

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.value
Frosty = BadRepr('☃')

def test_just_frosty():
    if False:
        return 10
    assert repr(st.just(Frosty)) == 'just(☃)'

def test_sampling_snowmen():
    if False:
        i = 10
        return i + 15
    assert repr(st.sampled_from((Frosty, 'hi'))) == "sampled_from((☃, 'hi'))"

def varargs(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    pass

@given(st.sampled_from(['✐', '✑', '✒', '✓', '✔', '✕', '✖', '✗', '✘', '✙', '✚', '✛', '✜', '✝', '✞', '✟', '✠', '✡', '✢', '✣']))
def test_sampled_from_bad_repr(c):
    if False:
        for i in range(10):
            print('nop')
    pass