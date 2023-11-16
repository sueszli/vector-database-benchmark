import numpy as np

class ExtraAssertionsMixin(object):

    def assertUnhashableCountEqual(self, data1, data2):
        if False:
            return 10
        'Assert that two containers have the same items, with special treatment\n    for numpy arrays.\n    '
        try:
            self.assertCountEqual(data1, data2)
        except (TypeError, ValueError):
            data1 = [self._to_hashable(d) for d in data1]
            data2 = [self._to_hashable(d) for d in data2]
            self.assertCountEqual(data1, data2)

    def _to_hashable(self, element):
        if False:
            i = 10
            return i + 15
        try:
            hash(element)
            return element
        except TypeError:
            pass
        if isinstance(element, list):
            return tuple((self._to_hashable(e) for e in element))
        if isinstance(element, dict):
            hashable_elements = []
            for (key, value) in sorted(element.items(), key=lambda t: hash(t[0])):
                hashable_elements.append((key, self._to_hashable(value)))
            return tuple(hashable_elements)
        if isinstance(element, np.ndarray):
            return element.tobytes()
        raise AssertionError('Encountered unhashable element: {}.'.format(element))