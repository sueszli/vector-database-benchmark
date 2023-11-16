import pytest
import t.skip
from celery.utils import term
from celery.utils.term import colored, fg

@t.skip.if_win32
class test_colored:

    @pytest.fixture(autouse=True)
    def preserve_encoding(self, patching):
        if False:
            return 10
        patching('sys.getdefaultencoding', 'utf-8')

    @pytest.mark.parametrize('name,color', [('black', term.BLACK), ('red', term.RED), ('green', term.GREEN), ('yellow', term.YELLOW), ('blue', term.BLUE), ('magenta', term.MAGENTA), ('cyan', term.CYAN), ('white', term.WHITE)])
    def test_colors(self, name, color):
        if False:
            i = 10
            return i + 15
        assert fg(30 + color) in str(colored().names[name]('foo'))

    @pytest.mark.parametrize('name', ['bold', 'underline', 'blink', 'reverse', 'bright', 'ired', 'igreen', 'iyellow', 'iblue', 'imagenta', 'icyan', 'iwhite', 'reset'])
    def test_modifiers(self, name):
        if False:
            for i in range(10):
                print('nop')
        assert str(getattr(colored(), name)('f'))

    def test_unicode(self):
        if False:
            while True:
                i = 10
        assert str(colored().green('∂bar'))
        assert colored().red('éefoo') + colored().green('∂bar')
        assert colored().red('foo').no_color() == 'foo'

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        assert repr(colored().blue('åfoo'))
        assert "''" in repr(colored())

    def test_more_unicode(self):
        if False:
            print('Hello World!')
        c = colored()
        s = c.red('foo', c.blue('bar'), c.green('baz'))
        assert s.no_color()
        c._fold_no_color(s, 'øfoo')
        c._fold_no_color('fooå', s)
        c = colored().red('åfoo')
        assert c._add(c, 'baræ') == '\x1b[1;31måfoo\x1b[0mbaræ'
        c2 = colored().blue('ƒƒz')
        c3 = c._add(c, c2)
        assert c3 == '\x1b[1;31måfoo\x1b[0m\x1b[1;34mƒƒz\x1b[0m'