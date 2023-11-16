import time
from .stringcache import StringIndex

class JsExecutionResult:

    def __init__(self, suite, statistics, errors, strings, basemillis=None, split_results=None, min_level=None, expand_keywords=None):
        if False:
            print('Hello World!')
        self.suite = suite
        self.strings = strings
        self.min_level = min_level
        self.data = self._get_data(statistics, errors, basemillis or 0, expand_keywords)
        self.split_results = split_results or []

    def _get_data(self, statistics, errors, basemillis, expand_keywords):
        if False:
            i = 10
            return i + 15
        return {'stats': statistics, 'errors': errors, 'baseMillis': basemillis, 'generated': int(time.time() * 1000) - basemillis, 'expand_keywords': expand_keywords}

    def remove_data_not_needed_in_report(self):
        if False:
            while True:
                i = 10
        self.data.pop('errors')
        remover = _KeywordRemover()
        self.suite = remover.remove_keywords(self.suite)
        (self.suite, self.strings) = remover.remove_unused_strings(self.suite, self.strings)

class _KeywordRemover:

    def remove_keywords(self, suite):
        if False:
            while True:
                i = 10
        return self._remove_keywords_from_suite(suite)

    def _remove_keywords_from_suite(self, suite):
        if False:
            return 10
        return suite[:6] + (self._remove_keywords_from_suites(suite[6]), self._remove_keywords_from_tests(suite[7]), (), suite[9])

    def _remove_keywords_from_suites(self, suites):
        if False:
            for i in range(10):
                print('nop')
        return tuple((self._remove_keywords_from_suite(s) for s in suites))

    def _remove_keywords_from_tests(self, tests):
        if False:
            while True:
                i = 10
        return tuple((self._remove_keywords_from_test(t) for t in tests))

    def _remove_keywords_from_test(self, test):
        if False:
            return 10
        return test[:-1] + ((),)

    def remove_unused_strings(self, model, strings):
        if False:
            while True:
                i = 10
        used = set(self._get_used_indices(model))
        remap = {}
        strings = tuple(self._get_used_strings(strings, used, remap))
        model = tuple(self._remap_string_indices(model, remap))
        return (model, strings)

    def _get_used_indices(self, model):
        if False:
            for i in range(10):
                print('nop')
        for item in model:
            if isinstance(item, StringIndex):
                yield item
            elif isinstance(item, tuple):
                for i in self._get_used_indices(item):
                    yield i

    def _get_used_strings(self, strings, used_indices, remap):
        if False:
            i = 10
            return i + 15
        offset = 0
        for (index, string) in enumerate(strings):
            if index in used_indices:
                remap[index] = index - offset
                yield string
            else:
                offset += 1

    def _remap_string_indices(self, model, remap):
        if False:
            while True:
                i = 10
        for item in model:
            if isinstance(item, StringIndex):
                yield remap[item]
            elif isinstance(item, tuple):
                yield tuple(self._remap_string_indices(item, remap))
            else:
                yield item