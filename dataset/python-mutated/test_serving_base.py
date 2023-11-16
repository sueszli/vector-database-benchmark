import random
import unittest
from odoo.tools import topological_sort

def sample(population):
    if False:
        for i in range(10):
            print('nop')
    return random.sample(population, random.randint(0, min(len(population), 5)))

class TestModulesLoading(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.mods = map(str, range(1000))

    def test_topological_sort(self):
        if False:
            i = 10
            return i + 15
        random.shuffle(self.mods)
        modules = [(k, sample(self.mods[:i])) for (i, k) in enumerate(self.mods)]
        random.shuffle(modules)
        ms = dict(modules)
        seen = set()
        sorted_modules = topological_sort(ms)
        for module in sorted_modules:
            deps = ms[module]
            self.assertGreaterEqual(seen, set(deps), 'Module %s (index %d), missing dependencies %s from loaded modules %s' % (module, sorted_modules.index(module), deps, seen))
            seen.add(module)