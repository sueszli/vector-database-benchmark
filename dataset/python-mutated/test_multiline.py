from django.test import SimpleTestCase
from ..utils import setup
multiline_string = '\nHello,\nboys.\nHow\nare\nyou\ngentlemen.\n'

class MultilineTests(SimpleTestCase):

    @setup({'multiline01': multiline_string})
    def test_multiline01(self):
        if False:
            i = 10
            return i + 15
        output = self.engine.render_to_string('multiline01')
        self.assertEqual(output, multiline_string)