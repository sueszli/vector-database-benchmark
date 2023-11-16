import functools
import unittest
from odoo.tools import frozendict
from odoo.tools.func import compose

class TestCompose(unittest.TestCase):

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        str_add = compose(str, lambda a, b: a + b)
        self.assertEqual(str_add(1, 2), '3')

    def test_decorator(self):
        if False:
            i = 10
            return i + 15
        ' ensure compose() can be partially applied as a decorator\n        '

        @functools.partial(compose, unicode)
        def mul(a, b):
            if False:
                i = 10
                return i + 15
            return a * b
        self.assertEqual(mul(5, 42), u'210')

class TestFrozendict(unittest.TestCase):

    def test_frozendict_immutable(self):
        if False:
            return 10
        ' Ensure that a frozendict is immutable. '
        vals = {'name': 'Joe', 'age': 42}
        frozen_vals = frozendict(vals)
        with self.assertRaises(Exception):
            frozen_vals['surname'] = 'Jack'
        with self.assertRaises(Exception):
            frozen_vals['name'] = 'Jack'
        with self.assertRaises(Exception):
            del frozen_vals['name']
        with self.assertRaises(Exception):
            frozen_vals.update({'surname': 'Jack'})
        with self.assertRaises(Exception):
            frozen_vals.update({'name': 'Jack'})
        with self.assertRaises(Exception):
            frozen_vals.setdefault('surname', 'Jack')
        with self.assertRaises(Exception):
            frozen_vals.pop('surname', 'Jack')
        with self.assertRaises(Exception):
            frozen_vals.pop('name', 'Jack')
        with self.assertRaises(Exception):
            frozen_vals.popitem()
        with self.assertRaises(Exception):
            frozen_vals.clear()

    def test_frozendict_hash(self):
        if False:
            return 10
        ' Ensure that a frozendict is hashable. '
        hash(frozendict({'name': 'Joe', 'age': 42}))
        hash(frozendict({'user_id': (42, 'Joe'), 'line_ids': [(0, 0, {'values': [42]})]}))