def fn(arg, *, kwarg='test', **kw):
    if False:
        print('Hello World!')
    assert arg == 1
    assert kwarg == 'testing'
    assert kw['foo'] == 'bar'
fn(1, kwarg='testing', foo='bar')