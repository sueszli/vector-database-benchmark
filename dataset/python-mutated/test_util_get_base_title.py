from test.picardtestcase import PicardTestCase
from picard.util import get_base_title_with_suffix

class GetBaseTitle(PicardTestCase):

    def test_base_title_0(self):
        if False:
            while True:
                i = 10
        test_title = 'title'
        title = get_base_title_with_suffix(test_title, '(copy)', '{title} ({count})')
        self.assertEqual(title, 'title')

    def test_base_title_1(self):
        if False:
            for i in range(10):
                print('nop')
        test_title = 'title (copy)'
        title = get_base_title_with_suffix(test_title, '(copy)', '{title} ({count})')
        self.assertEqual(title, 'title')

    def test_base_title_2(self):
        if False:
            i = 10
            return i + 15
        test_title = 'title (copy) (1)'
        title = get_base_title_with_suffix(test_title, '(copy)', '{title} ({count})')
        self.assertEqual(title, 'title')

    def test_base_title_3(self):
        if False:
            for i in range(10):
                print('nop')
        test_title = 'title (copy)(1)'
        title = get_base_title_with_suffix(test_title, '(copy)', '{title} ({count})')
        self.assertEqual(title, test_title)

    def test_base_title_4(self):
        if False:
            print('Hello World!')
        test_title = 'title (copy)()'
        title = get_base_title_with_suffix(test_title, '(copy)', '{title} ({count})')
        self.assertEqual(title, test_title)

    def test_base_title_5(self):
        if False:
            while True:
                i = 10
        test_title = 'title (copy) ()'
        title = get_base_title_with_suffix(test_title, '(copy)', '{title} ({count})')
        self.assertEqual(title, 'title')

    def test_base_title_6(self):
        if False:
            print('Hello World!')
        test_title = 'title (copy) (x)'
        title = get_base_title_with_suffix(test_title, '(copy)', '{title} ({count})')
        self.assertEqual(title, test_title)

    def test_base_title_7(self):
        if False:
            while True:
                i = 10
        test_title = 'title (copy) (1)x'
        title = get_base_title_with_suffix(test_title, '(copy)', '{title} ({count})')
        self.assertEqual(title, test_title)

    def test_base_title_8(self):
        if False:
            while True:
                i = 10
        test_title = 'title (copy) (1)'
        title = get_base_title_with_suffix(test_title, '(c?py)', '{title} ({count})')
        self.assertEqual(title, 'title (copy)')

    def test_base_title_9(self):
        if False:
            while True:
                i = 10
        test_title = 'title (copy)'
        title = get_base_title_with_suffix(test_title, '(copy)', '({count}) {title}')
        self.assertEqual(title, 'title')

    def test_base_title_10(self):
        if False:
            for i in range(10):
                print('nop')
        test_title = 'title (copy) (1)'
        title = get_base_title_with_suffix(test_title, '(copy)', '({count}) {title}')
        self.assertEqual(title, test_title)

    def test_base_title_11(self):
        if False:
            return 10
        test_title = '(1) title (copy)'
        title = get_base_title_with_suffix(test_title, '(copy)', '({count}) {title}')
        self.assertEqual(title, 'title')

    def test_base_title_12(self):
        if False:
            print('Hello World!')
        test_title = '(1) title (copy) (1)'
        title = get_base_title_with_suffix(test_title, '(copy)', '({count}) {title}')
        self.assertEqual(title, 'title (copy) (1)')