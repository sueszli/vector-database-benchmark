import vaex

def test_apply():
    if False:
        for i in range(10):
            print('nop')
    df = vaex.from_arrays(x=[1, 2, 3])
    assert df.x.apply(lambda x: x + 1).values.tolist() == [2, 3, 4]

def test_apply_vectorized():
    if False:
        while True:
            i = 10
    df = vaex.from_arrays(x=[1, 2, 3])
    assert df.x.apply(lambda x: x + 1, vectorize=True).values.tolist() == [2, 3, 4]

def test_apply_select():
    if False:
        return 10
    x = [1, 2, 3, 4, 5]
    df = vaex.from_arrays(x=x)
    df['x_ind'] = df.x.apply(lambda w: w > 3)
    df.state_get()
    df.select('x_ind')
    assert 2 == df.selected_length()

def test_apply_with_invalid_identifier():
    if False:
        print('Hello World!')
    df = vaex.from_dict({'#': [1], 'with space': [2]})

    def add(a, b):
        if False:
            while True:
                i = 10
        return a + b
    assert df.apply(add, arguments=[df['#'], df['with space']]).tolist() == [3]