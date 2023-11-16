from feeluown.server import dslv1
from feeluown.server import Request

def test_request():
    if False:
        return 10
    req = Request(cmd='play', cmd_args=['fuo://x'])
    assert 'play fuo://x' in dslv1.unparse(req)