from weakref import WeakKeyDictionary
from hypothesis.control import note
from hypothesis.errors import InvalidState
from hypothesis.internal.reflection import convert_positional_arguments, nicerepr, proxies, repr_call
from hypothesis.strategies._internal.strategies import SearchStrategy

class FunctionStrategy(SearchStrategy):
    supports_find = False

    def __init__(self, like, returns, pure):
        if False:
            return 10
        super().__init__()
        self.like = like
        self.returns = returns
        self.pure = pure
        self._cache = WeakKeyDictionary()

    def calc_is_empty(self, recur):
        if False:
            print('Hello World!')
        return recur(self.returns)

    def do_draw(self, data):
        if False:
            i = 10
            return i + 15

        @proxies(self.like)
        def inner(*args, **kwargs):
            if False:
                return 10
            if data.frozen:
                raise InvalidState(f'This generated {nicerepr(self.like)} function can only be called within the scope of the @given that created it.')
            if self.pure:
                (args, kwargs) = convert_positional_arguments(self.like, args, kwargs)
                key = (args, frozenset(kwargs.items()))
                cache = self._cache.setdefault(inner, {})
                if key not in cache:
                    cache[key] = data.draw(self.returns)
                    rep = repr_call(self.like, args, kwargs, reorder=False)
                    note(f'Called function: {rep} -> {cache[key]!r}')
                return cache[key]
            else:
                val = data.draw(self.returns)
                rep = repr_call(self.like, args, kwargs, reorder=False)
                note(f'Called function: {rep} -> {val!r}')
                return val
        return inner