from hypothesis.internal.conjecture import utils as cu
from hypothesis.strategies._internal.strategies import SearchStrategy
FEATURE_LABEL = cu.calc_label_from_name('feature flag')

class FeatureFlags:
    """Object that can be used to control a number of feature flags for a
    given test run.

    This enables an approach to data generation called swarm testing (
    see Groce, Alex, et al. "Swarm testing." Proceedings of the 2012
    International Symposium on Software Testing and Analysis. ACM, 2012), in
    which generation is biased by selectively turning some features off for
    each test case generated. When there are many interacting features this can
    find bugs that a pure generation strategy would otherwise have missed.

    FeatureFlags are designed to "shrink open", so that during shrinking they
    become less restrictive. This allows us to potentially shrink to smaller
    test cases that were forbidden during the generation phase because they
    required disabled features.
    """

    def __init__(self, data=None, enabled=(), disabled=()):
        if False:
            while True:
                i = 10
        self.__data = data
        self.__is_disabled = {}
        for f in enabled:
            self.__is_disabled[f] = False
        for f in disabled:
            self.__is_disabled[f] = True
        if self.__data is not None:
            self.__p_disabled = data.draw_bits(8) / 255.0
        else:
            self.__p_disabled = 0.0

    def is_enabled(self, name):
        if False:
            i = 10
            return i + 15
        'Tests whether the feature named ``name`` should be enabled on this\n        test run.'
        if self.__data is None or self.__data.frozen:
            return not self.__is_disabled.get(name, False)
        data = self.__data
        data.start_example(label=FEATURE_LABEL)
        is_disabled = cu.biased_coin(self.__data, self.__p_disabled, forced=self.__is_disabled.get(name))
        self.__is_disabled[name] = is_disabled
        data.stop_example()
        return not is_disabled

    def __repr__(self):
        if False:
            return 10
        enabled = []
        disabled = []
        for (name, is_disabled) in self.__is_disabled.items():
            if is_disabled:
                disabled.append(name)
            else:
                enabled.append(name)
        return f'FeatureFlags(enabled={enabled!r}, disabled={disabled!r})'

class FeatureStrategy(SearchStrategy):

    def do_draw(self, data):
        if False:
            i = 10
            return i + 15
        return FeatureFlags(data)