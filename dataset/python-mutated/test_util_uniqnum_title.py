from test.picardtestcase import PicardTestCase
from picard.util import _regex_numbered_title_fmt, unique_numbered_title

class RegexNumberedTitleFmt(PicardTestCase):

    def test_1(self):
        if False:
            i = 10
            return i + 15
        fmt = ''
        result = _regex_numbered_title_fmt(fmt, 'TITLE', 'COUNT')
        self.assertEqual(result, '')

    def test_2(self):
        if False:
            print('Hello World!')
        fmt = '{title} {count}'
        result = _regex_numbered_title_fmt(fmt, 'TITLE', 'COUNT')
        self.assertEqual(result, 'TITLE(?:\\ COUNT)?')

    def test_3(self):
        if False:
            print('Hello World!')
        fmt = 'x {count}  {title} y'
        result = _regex_numbered_title_fmt(fmt, 'TITLE', 'COUNT')
        self.assertEqual(result, '(?:x\\ COUNT\\ \\ )?TITLE y')

    def test_4(self):
        if False:
            while True:
                i = 10
        fmt = 'x {title}{count} y'
        result = _regex_numbered_title_fmt(fmt, 'TITLE', 'COUNT')
        self.assertEqual(result, 'x TITLE(?:COUNT\\ y)?')

class UniqueNumberedTitle(PicardTestCase):

    def test_existing_titles_0(self):
        if False:
            for i in range(10):
                print('nop')
        title = unique_numbered_title('title', [], fmt='{title} ({count})')
        self.assertEqual(title, 'title (1)')

    def test_existing_titles_1(self):
        if False:
            print('Hello World!')
        title = unique_numbered_title('title', ['title'], fmt='{title} ({count})')
        self.assertEqual(title, 'title (2)')

    def test_existing_titles_2(self):
        if False:
            i = 10
            return i + 15
        title = unique_numbered_title('title', ['title', 'title (2)'], fmt='{title} ({count})')
        self.assertEqual(title, 'title (3)')

    def test_existing_titles_3(self):
        if False:
            print('Hello World!')
        title = unique_numbered_title('title', ['title (1)', 'title (2)'], fmt='{title} ({count})')
        self.assertEqual(title, 'title (3)')

    def test_existing_titles_4(self):
        if False:
            i = 10
            return i + 15
        title = unique_numbered_title('title', ['title', 'title'], fmt='{title} ({count})')
        self.assertEqual(title, 'title (3)')

    def test_existing_titles_5(self):
        if False:
            i = 10
            return i + 15
        title = unique_numbered_title('title', ['x title', 'title y'], fmt='{title} ({count})')
        self.assertEqual(title, 'title (1)')

    def test_existing_titles_6(self):
        if False:
            i = 10
            return i + 15
        title = unique_numbered_title('title', ['title (n)'], fmt='{title} ({count})')
        self.assertEqual(title, 'title (1)')

    def test_existing_titles_7(self):
        if False:
            return 10
        title = unique_numbered_title('title', ['title ()'], fmt='{title} ({count})')
        self.assertEqual(title, 'title (1)')

    def test_existing_titles_8(self):
        if False:
            for i in range(10):
                print('nop')
        title = unique_numbered_title('title', ['title(2)'], fmt='{title} ({count})')
        self.assertEqual(title, 'title (1)')

class UniqueNumberedTitleFmt(PicardTestCase):

    def test_existing_titles_0(self):
        if False:
            while True:
                i = 10
        title = unique_numbered_title('title', [], fmt='({count}) {title}')
        self.assertEqual(title, '(1) title')

    def test_existing_titles_1(self):
        if False:
            return 10
        title = unique_numbered_title('title', ['title'], fmt='({count}) {title}')
        self.assertEqual(title, '(2) title')

    def test_existing_titles_2(self):
        if False:
            while True:
                i = 10
        title = unique_numbered_title('title', ['title', '(2) title'], fmt='({count}) {title}')
        self.assertEqual(title, '(3) title')

    def test_existing_titles_3(self):
        if False:
            print('Hello World!')
        title = unique_numbered_title('title', ['(1) title', '(2) title'], fmt='({count}) {title}')
        self.assertEqual(title, '(3) title')

    def test_existing_titles_4(self):
        if False:
            for i in range(10):
                print('nop')
        title = unique_numbered_title('title', ['title', 'title'], fmt='({count}) {title}')
        self.assertEqual(title, '(3) title')

    def test_existing_titles_5(self):
        if False:
            return 10
        title = unique_numbered_title('title', ['x title', 'title y'], fmt='({count}) {title}')
        self.assertEqual(title, '(1) title')

    def test_existing_titles_6(self):
        if False:
            print('Hello World!')
        title = unique_numbered_title('title', ['(n) title'], fmt='({count}) {title}')
        self.assertEqual(title, '(1) title')

    def test_existing_titles_7(self):
        if False:
            while True:
                i = 10
        title = unique_numbered_title('title', ['() title'], fmt='({count}) {title}')
        self.assertEqual(title, '(1) title')

    def test_existing_titles_8(self):
        if False:
            for i in range(10):
                print('nop')
        title = unique_numbered_title('title', ['(2)title'], fmt='({count}) {title}')
        self.assertEqual(title, '(1) title')