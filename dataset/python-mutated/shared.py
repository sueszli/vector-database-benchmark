from hypothesis.strategies._internal import SearchStrategy
SHARED_STRATEGY_ATTRIBUTE = '_hypothesis_shared_strategies'

class SharedStrategy(SearchStrategy):

    def __init__(self, base, key=None):
        if False:
            return 10
        self.key = key
        self.base = base

    @property
    def supports_find(self):
        if False:
            for i in range(10):
                print('nop')
        return self.base.supports_find

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self.key is not None:
            return f'shared({self.base!r}, key={self.key!r})'
        else:
            return f'shared({self.base!r})'

    def do_draw(self, data):
        if False:
            i = 10
            return i + 15
        if not hasattr(data, SHARED_STRATEGY_ATTRIBUTE):
            setattr(data, SHARED_STRATEGY_ATTRIBUTE, {})
        sharing = getattr(data, SHARED_STRATEGY_ATTRIBUTE)
        key = self.key or self
        if key not in sharing:
            sharing[key] = data.draw(self.base)
        return sharing[key]