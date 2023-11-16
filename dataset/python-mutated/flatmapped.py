from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.strategies._internal.strategies import SearchStrategy, check_strategy

class FlatMapStrategy(SearchStrategy):

    def __init__(self, strategy, expand):
        if False:
            return 10
        super().__init__()
        self.flatmapped_strategy = strategy
        self.expand = expand

    def calc_is_empty(self, recur):
        if False:
            while True:
                i = 10
        return recur(self.flatmapped_strategy)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, '_cached_repr'):
            self._cached_repr = f'{self.flatmapped_strategy!r}.flatmap({get_pretty_function_description(self.expand)})'
        return self._cached_repr

    def do_draw(self, data):
        if False:
            for i in range(10):
                print('nop')
        source = data.draw(self.flatmapped_strategy)
        expanded_source = self.expand(source)
        check_strategy(expanded_source)
        return data.draw(expanded_source)

    @property
    def branches(self):
        if False:
            i = 10
            return i + 15
        return [FlatMapStrategy(strategy=strategy, expand=self.expand) for strategy in self.flatmapped_strategy.branches]