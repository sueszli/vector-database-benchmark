import contextlib
import functools
from unittest import mock
from twisted.internet import defer

def _dot_lookup(thing, comp, import_path):
    if False:
        return 10
    try:
        return getattr(thing, comp)
    except AttributeError:
        __import__(import_path)
        return getattr(thing, comp)

def _importer(target):
    if False:
        return 10
    components = target.split('.')
    import_path = components.pop(0)
    thing = __import__(import_path)
    for comp in components:
        import_path += f'.{comp}'
        thing = _dot_lookup(thing, comp, import_path)
    return thing

def _get_target(target):
    if False:
        print('Hello World!')
    try:
        (target, attribute) = target.rsplit('.', 1)
    except (TypeError, ValueError) as e:
        raise TypeError(f'Need a valid target to patch. You supplied: {repr(target)}') from e
    return (_importer(target), attribute)

class DelayWrapper:

    def __init__(self):
        if False:
            return 10
        self._deferreds = []

    def add_new(self):
        if False:
            print('Hello World!')
        d = defer.Deferred()
        self._deferreds.append(d)
        return d

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._deferreds)

    def fire(self):
        if False:
            print('Hello World!')
        deferreds = self._deferreds
        self._deferreds = []
        for d in deferreds:
            d.callback(None)

@contextlib.contextmanager
def patchForDelay(target_name):
    if False:
        return 10

    class Default:
        pass
    default = Default()
    (target, attribute) = _get_target(target_name)
    original = getattr(target, attribute, default)
    if original is default:
        raise RuntimeError(f'Could not find name {target_name}')
    if not callable(original):
        raise RuntimeError(f'{target_name} is not callable')
    delay = DelayWrapper()

    @functools.wraps(original)
    @defer.inlineCallbacks
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        yield delay.add_new()
        return (yield original(*args, **kwargs))
    with mock.patch(target_name, new=wrapper):
        try:
            yield delay
        finally:
            delay.fire()