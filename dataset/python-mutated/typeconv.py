try:
    from numba.core.typeconv import _typeconv
except ImportError as e:
    base_url = 'https://numba.readthedocs.io/en/stable'
    dev_url = f'{base_url}/developer/contributing.html'
    user_url = f'{base_url}/user/faq.html#numba-could-not-be-imported'
    dashes = '-' * 80
    msg = f"Numba could not be imported.\n{dashes}\nIf you are seeing this message and are undertaking Numba development work, you may need to rebuild Numba.\nPlease see the development set up guide:\n\n{dev_url}.\n\n{dashes}\nIf you are not working on Numba development, the original error was: '{str(e)}'.\nFor help, please visit:\n\n{user_url}\n"
    raise ImportError(msg)
from numba.core.typeconv import castgraph, Conversion
from numba.core import types

class TypeManager(object):
    _conversion_codes = {Conversion.safe: ord('s'), Conversion.unsafe: ord('u'), Conversion.promote: ord('p')}

    def __init__(self):
        if False:
            return 10
        self._ptr = _typeconv.new_type_manager()
        self._types = set()

    def select_overload(self, sig, overloads, allow_unsafe, exact_match_required):
        if False:
            for i in range(10):
                print('nop')
        sig = [t._code for t in sig]
        overloads = [[t._code for t in s] for s in overloads]
        return _typeconv.select_overload(self._ptr, sig, overloads, allow_unsafe, exact_match_required)

    def check_compatible(self, fromty, toty):
        if False:
            i = 10
            return i + 15
        if not isinstance(toty, types.Type):
            raise ValueError("Specified type '%s' (%s) is not a Numba type" % (toty, type(toty)))
        name = _typeconv.check_compatible(self._ptr, fromty._code, toty._code)
        conv = Conversion[name] if name is not None else None
        assert conv is not Conversion.nil
        return conv

    def set_compatible(self, fromty, toty, by):
        if False:
            print('Hello World!')
        code = self._conversion_codes[by]
        _typeconv.set_compatible(self._ptr, fromty._code, toty._code, code)
        self._types.add(fromty)
        self._types.add(toty)

    def set_promote(self, fromty, toty):
        if False:
            i = 10
            return i + 15
        self.set_compatible(fromty, toty, Conversion.promote)

    def set_unsafe_convert(self, fromty, toty):
        if False:
            for i in range(10):
                print('nop')
        self.set_compatible(fromty, toty, Conversion.unsafe)

    def set_safe_convert(self, fromty, toty):
        if False:
            return 10
        self.set_compatible(fromty, toty, Conversion.safe)

    def get_pointer(self):
        if False:
            for i in range(10):
                print('nop')
        return _typeconv.get_pointer(self._ptr)

class TypeCastingRules(object):
    """
    A helper for establishing type casting rules.
    """

    def __init__(self, tm):
        if False:
            while True:
                i = 10
        self._tm = tm
        self._tg = castgraph.TypeGraph(self._cb_update)

    def promote(self, a, b):
        if False:
            print('Hello World!')
        '\n        Set `a` can promote to `b`\n        '
        self._tg.promote(a, b)

    def unsafe(self, a, b):
        if False:
            while True:
                i = 10
        '\n        Set `a` can unsafe convert to `b`\n        '
        self._tg.unsafe(a, b)

    def safe(self, a, b):
        if False:
            while True:
                i = 10
        '\n        Set `a` can safe convert to `b`\n        '
        self._tg.safe(a, b)

    def promote_unsafe(self, a, b):
        if False:
            return 10
        '\n        Set `a` can promote to `b` and `b` can unsafe convert to `a`\n        '
        self.promote(a, b)
        self.unsafe(b, a)

    def safe_unsafe(self, a, b):
        if False:
            return 10
        '\n        Set `a` can safe convert to `b` and `b` can unsafe convert to `a`\n        '
        self._tg.safe(a, b)
        self._tg.unsafe(b, a)

    def unsafe_unsafe(self, a, b):
        if False:
            return 10
        '\n        Set `a` can unsafe convert to `b` and `b` can unsafe convert to `a`\n        '
        self._tg.unsafe(a, b)
        self._tg.unsafe(b, a)

    def _cb_update(self, a, b, rel):
        if False:
            print('Hello World!')
        '\n        Callback for updating.\n        '
        if rel == Conversion.promote:
            self._tm.set_promote(a, b)
        elif rel == Conversion.safe:
            self._tm.set_safe_convert(a, b)
        elif rel == Conversion.unsafe:
            self._tm.set_unsafe_convert(a, b)
        else:
            raise AssertionError(rel)