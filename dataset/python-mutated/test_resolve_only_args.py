from .. import deprecated
from ..resolve_only_args import resolve_only_args

def test_resolve_only_args(mocker):
    if False:
        while True:
            i = 10
    mocker.patch.object(deprecated, 'warn_deprecation')

    def resolver(root, **args):
        if False:
            for i in range(10):
                print('nop')
        return (root, args)
    wrapped_resolver = resolve_only_args(resolver)
    assert deprecated.warn_deprecation.called
    result = wrapped_resolver(1, 2, a=3)
    assert result == (1, {'a': 3})