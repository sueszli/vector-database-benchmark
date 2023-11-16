from common import *

class MyFunc:

    def __init__(self, a):
        if False:
            return 10
        self.a = a

    def __call__(self, x, y):
        if False:
            return 10
        return self.a * x + y

@pytest.fixture()
def df():
    if False:
        print('Hello World!')
    df = vaex.from_scalars(x=1, y=2, z=3)
    df['u'] = df.x + df.y
    df['v'] = np.sin(df.u / df.y)
    myfunc = MyFunc(2)
    df.add_function('myfunc', myfunc)
    df['w'] = df.func.myfunc(df.v, df.z)
    df['b'] = df.x
    df['c'] = df.y
    df['d'] = df.c
    return df

def test_dependencies():
    if False:
        return 10
    df = vaex.from_scalars(x=1, y=2, z=3)
    df['u'] = df.x + df.y
    u_dep = ['u', None, None, [['(x + y)', '+', None, ['x', 'y']]]]
    assert df.u._graph() == u_dep
    df['v'] = np.sin(df.u / df.y)
    v_dep = ['v', None, None, [['sin((u / y))', 'sin', None, [['(u / y)', '/', None, [u_dep, 'y']]]]]]
    assert df.v._graph() == v_dep
    myfunc = MyFunc(2)
    df.add_function('myfunc', myfunc)
    df['w'] = df.func.myfunc(df.v, df.z)
    assert df.w._graph() == ['w', None, None, [['myfunc(v, z)', 'myfunc', myfunc, [v_dep, 'z']]]]
    print(df.w._graph())

def test_graphviz(df):
    if False:
        print('Hello World!')
    assert df._graphviz() is not None

def test_root_nodes(df):
    if False:
        while True:
            i = 10
    assert df._root_nodes() == ['w', 'b', 'd']