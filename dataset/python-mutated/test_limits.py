from hypothesis import given, settings, strategies as st

def test_max_examples_are_respected():
    if False:
        print('Hello World!')
    counter = [0]

    @given(st.random_module(), st.integers())
    @settings(max_examples=100)
    def test(rnd, i):
        if False:
            print('Hello World!')
        counter[0] += 1
    test()
    assert counter == [100]