import pickle
from boltons.typeutils import make_sentinel
NOT_SET = make_sentinel('not_set', var_name='NOT_SET')

def test_sentinel_falsiness():
    if False:
        print('Hello World!')
    assert not NOT_SET

def test_sentinel_pickle():
    if False:
        i = 10
        return i + 15
    assert pickle.dumps(NOT_SET)