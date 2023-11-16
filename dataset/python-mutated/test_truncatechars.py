from django.test import SimpleTestCase
from ..utils import setup

class TruncatecharsTests(SimpleTestCase):

    @setup({'truncatechars01': '{{ a|truncatechars:3 }}'})
    def test_truncatechars01(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('truncatechars01', {'a': 'Testing, testing'})
        self.assertEqual(output, 'Te…')

    @setup({'truncatechars02': '{{ a|truncatechars:7 }}'})
    def test_truncatechars02(self):
        if False:
            return 10
        output = self.engine.render_to_string('truncatechars02', {'a': 'Testing'})
        self.assertEqual(output, 'Testing')

    @setup({'truncatechars03': "{{ a|truncatechars:'e' }}"})
    def test_fail_silently_incorrect_arg(self):
        if False:
            return 10
        output = self.engine.render_to_string('truncatechars03', {'a': 'Testing, testing'})
        self.assertEqual(output, 'Testing, testing')