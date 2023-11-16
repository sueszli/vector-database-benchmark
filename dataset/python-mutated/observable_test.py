"""Tests for the Observable mixin class."""
import logging
import unittest
from typing import List
from typing import Optional
from apache_beam.coders import observable

class ObservableMixinTest(unittest.TestCase):
    observed_count = 0
    observed_sum = 0
    observed_keys = []

    def observer(self, value, key=None):
        if False:
            return 10
        self.observed_count += 1
        self.observed_sum += value
        self.observed_keys.append(key)

    def test_observable(self):
        if False:
            return 10

        class Watched(observable.ObservableMixin):

            def __iter__(self):
                if False:
                    print('Hello World!')
                for i in (1, 4, 3):
                    self.notify_observers(i, key='a%d' % i)
                    yield i
        watched = Watched()
        watched.register_observer(lambda v, key: self.observer(v, key=key))
        for _ in watched:
            pass
        self.assertEqual(3, self.observed_count)
        self.assertEqual(8, self.observed_sum)
        self.assertEqual(['a1', 'a3', 'a4'], sorted(self.observed_keys))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()