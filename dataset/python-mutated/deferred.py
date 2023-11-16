import inspect
from hypothesis.errors import InvalidArgument
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.strategies._internal.strategies import SearchStrategy, check_strategy

class DeferredStrategy(SearchStrategy):
    """A strategy which may be used before it is fully defined."""

    def __init__(self, definition):
        if False:
            return 10
        super().__init__()
        self.__wrapped_strategy = None
        self.__in_repr = False
        self.__definition = definition

    @property
    def wrapped_strategy(self):
        if False:
            for i in range(10):
                print('nop')
        if self.__wrapped_strategy is None:
            if not inspect.isfunction(self.__definition):
                raise InvalidArgument(f'Expected definition to be a function but got {self.__definition!r} of type {type(self.__definition).__name__} instead.')
            result = self.__definition()
            if result is self:
                raise InvalidArgument('Cannot define a deferred strategy to be itself')
            check_strategy(result, 'definition()')
            self.__wrapped_strategy = result
            self.__definition = None
        return self.__wrapped_strategy

    @property
    def branches(self):
        if False:
            for i in range(10):
                print('nop')
        return self.wrapped_strategy.branches

    @property
    def supports_find(self):
        if False:
            for i in range(10):
                print('nop')
        return self.wrapped_strategy.supports_find

    def calc_label(self):
        if False:
            print('Hello World!')
        "Deferred strategies don't have a calculated label, because we would\n        end up having to calculate the fixed point of some hash function in\n        order to calculate it when they recursively refer to themself!\n\n        The label for the wrapped strategy will still appear because it\n        will be passed to draw.\n        "
        return self.class_label

    def calc_is_empty(self, recur):
        if False:
            while True:
                i = 10
        return recur(self.wrapped_strategy)

    def calc_has_reusable_values(self, recur):
        if False:
            return 10
        return recur(self.wrapped_strategy)

    def __repr__(self):
        if False:
            print('Hello World!')
        if self.__wrapped_strategy is not None:
            if self.__in_repr:
                return f'(deferred@{id(self)!r})'
            try:
                self.__in_repr = True
                return repr(self.__wrapped_strategy)
            finally:
                self.__in_repr = False
        else:
            description = get_pretty_function_description(self.__definition)
            return f'deferred({description})'

    def do_draw(self, data):
        if False:
            return 10
        return data.draw(self.wrapped_strategy)