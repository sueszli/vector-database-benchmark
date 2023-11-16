from django.test import SimpleTestCase
from ..utils import setup

class ListIndexTests(SimpleTestCase):

    @setup({'list-index01': '{{ var.1 }}'})
    def test_list_index01(self):
        if False:
            i = 10
            return i + 15
        '\n        List-index syntax allows a template to access a certain item of a\n        subscriptable object.\n        '
        output = self.engine.render_to_string('list-index01', {'var': ['first item', 'second item']})
        self.assertEqual(output, 'second item')

    @setup({'list-index02': '{{ var.5 }}'})
    def test_list_index02(self):
        if False:
            while True:
                i = 10
        '\n        Fail silently when the list index is out of range.\n        '
        output = self.engine.render_to_string('list-index02', {'var': ['first item', 'second item']})
        if self.engine.string_if_invalid:
            self.assertEqual(output, 'INVALID')
        else:
            self.assertEqual(output, '')

    @setup({'list-index03': '{{ var.1 }}'})
    def test_list_index03(self):
        if False:
            return 10
        '\n        Fail silently when the list index is out of range.\n        '
        output = self.engine.render_to_string('list-index03', {'var': None})
        if self.engine.string_if_invalid:
            self.assertEqual(output, 'INVALID')
        else:
            self.assertEqual(output, '')

    @setup({'list-index04': '{{ var.1 }}'})
    def test_list_index04(self):
        if False:
            i = 10
            return i + 15
        '\n        Fail silently when variable is a dict without the specified key.\n        '
        output = self.engine.render_to_string('list-index04', {'var': {}})
        if self.engine.string_if_invalid:
            self.assertEqual(output, 'INVALID')
        else:
            self.assertEqual(output, '')

    @setup({'list-index05': '{{ var.1 }}'})
    def test_list_index05(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Dictionary lookup wins out when dict's key is a string.\n        "
        output = self.engine.render_to_string('list-index05', {'var': {'1': 'hello'}})
        self.assertEqual(output, 'hello')

    @setup({'list-index06': '{{ var.1 }}'})
    def test_list_index06(self):
        if False:
            print('Hello World!')
        "\n        But list-index lookup wins out when dict's key is an int, which\n        behind the scenes is really a dictionary lookup (for a dict)\n        after converting the key to an int.\n        "
        output = self.engine.render_to_string('list-index06', {'var': {1: 'hello'}})
        self.assertEqual(output, 'hello')

    @setup({'list-index07': '{{ var.1 }}'})
    def test_list_index07(self):
        if False:
            while True:
                i = 10
        '\n        Dictionary lookup wins out when there is a string and int version\n        of the key.\n        '
        output = self.engine.render_to_string('list-index07', {'var': {'1': 'hello', 1: 'world'}})
        self.assertEqual(output, 'hello')