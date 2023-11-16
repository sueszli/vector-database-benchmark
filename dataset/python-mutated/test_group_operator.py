from odoo.tests import common

class TestGroupBooleans(common.TransactionCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestGroupBooleans, self).setUp()
        self.Model = self.env['test_read_group.aggregate.boolean']

    def test_no_value(self):
        if False:
            for i in range(10):
                print('nop')
        groups = self.Model.read_group(domain=[], fields=['key', 'bool_and', 'bool_or', 'bool_array'], groupby=['key'])
        self.assertEqual([], groups)

    def test_agg_and(self):
        if False:
            while True:
                i = 10
        self.Model.create({'key': 1, 'bool_and': True})
        self.Model.create({'key': 1, 'bool_and': True})
        self.Model.create({'key': 2, 'bool_and': True})
        self.Model.create({'key': 2, 'bool_and': False})
        self.Model.create({'key': 3, 'bool_and': False})
        self.Model.create({'key': 3, 'bool_and': False})
        groups = self.Model.read_group(domain=[], fields=['key', 'bool_and'], groupby=['key'])
        self.assertEqual([{'key_count': 2, '__domain': [('key', '=', 1)], 'key': 1, 'bool_and': True}, {'key_count': 2, '__domain': [('key', '=', 2)], 'key': 2, 'bool_and': False}, {'key_count': 2, '__domain': [('key', '=', 3)], 'key': 3, 'bool_and': False}], groups)

    def test_agg_or(self):
        if False:
            for i in range(10):
                print('nop')
        self.Model.create({'key': 1, 'bool_or': True})
        self.Model.create({'key': 1, 'bool_or': True})
        self.Model.create({'key': 2, 'bool_or': True})
        self.Model.create({'key': 2, 'bool_or': False})
        self.Model.create({'key': 3, 'bool_or': False})
        self.Model.create({'key': 3, 'bool_or': False})
        groups = self.Model.read_group(domain=[], fields=['key', 'bool_or'], groupby=['key'])
        self.assertEqual([{'key_count': 2, '__domain': [('key', '=', 1)], 'key': 1, 'bool_or': True}, {'key_count': 2, '__domain': [('key', '=', 2)], 'key': 2, 'bool_or': True}, {'key_count': 2, '__domain': [('key', '=', 3)], 'key': 3, 'bool_or': False}], groups)

    def test_agg_array(self):
        if False:
            print('Hello World!')
        self.Model.create({'key': 1, 'bool_array': True})
        self.Model.create({'key': 1, 'bool_array': True})
        self.Model.create({'key': 2, 'bool_array': True})
        self.Model.create({'key': 2, 'bool_array': False})
        self.Model.create({'key': 3, 'bool_array': False})
        self.Model.create({'key': 3, 'bool_array': False})
        groups = self.Model.read_group(domain=[], fields=['key', 'bool_array'], groupby=['key'])
        self.assertEqual([{'key_count': 2, '__domain': [('key', '=', 1)], 'key': 1, 'bool_array': [True, True]}, {'key_count': 2, '__domain': [('key', '=', 2)], 'key': 2, 'bool_array': [True, False]}, {'key_count': 2, '__domain': [('key', '=', 3)], 'key': 3, 'bool_array': [False, False]}], groups)

    def test_group_by_aggregable(self):
        if False:
            return 10
        self.Model.create({'bool_and': False, 'key': 1, 'bool_array': True})
        self.Model.create({'bool_and': False, 'key': 2, 'bool_array': True})
        self.Model.create({'bool_and': False, 'key': 2, 'bool_array': True})
        self.Model.create({'bool_and': True, 'key': 2, 'bool_array': True})
        self.Model.create({'bool_and': True, 'key': 3, 'bool_array': True})
        self.Model.create({'bool_and': True, 'key': 3, 'bool_array': True})
        groups = self.Model.read_group(domain=[], fields=['key', 'bool_and', 'bool_array'], groupby=['bool_and', 'key'], lazy=False)
        self.assertEqual([{'bool_and': False, 'key': 1, 'bool_array': [True], '__count': 1, '__domain': ['&', ('bool_and', '=', False), ('key', '=', 1)]}, {'bool_and': False, 'key': 2, 'bool_array': [True, True], '__count': 2, '__domain': ['&', ('bool_and', '=', False), ('key', '=', 2)]}, {'bool_and': True, 'key': 2, 'bool_array': [True], '__count': 1, '__domain': ['&', ('bool_and', '=', True), ('key', '=', 2)]}, {'bool_and': True, 'key': 3, 'bool_array': [True, True], '__count': 2, '__domain': ['&', ('bool_and', '=', True), ('key', '=', 3)]}], groups)