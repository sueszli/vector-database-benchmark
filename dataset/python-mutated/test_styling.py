import click
import pytest
from click import style
from doitlive import TermString, TTY

class TestTermString:

    @pytest.fixture
    def ts(self):
        if False:
            return 10
        return TermString('foo')

    @pytest.fixture
    def ts_blank(self):
        if False:
            while True:
                i = 10
        return TermString('')

    def test_str(self, ts):
        if False:
            while True:
                i = 10
        assert str(ts) == 'foo'

    @pytest.mark.parametrize('color', click.termui._ansi_colors)
    def test_color(self, color, ts):
        if False:
            i = 10
            return i + 15
        colored = getattr(ts, color)
        assert isinstance(colored, TermString)
        assert str(colored) == style('foo', fg=color)

    def test_bold(self, ts):
        if False:
            while True:
                i = 10
        assert str(ts.bold) == style('foo', bold=True)

    def test_blink(self, ts):
        if False:
            i = 10
            return i + 15
        assert str(ts.blink) == style('foo', blink=True)

    def test_dim(self, ts):
        if False:
            for i in range(10):
                print('nop')
        assert str(ts.dim) == style('foo', dim=True)

    def test_underlined(self, ts):
        if False:
            print('Hello World!')
        assert str(ts.underlined) == style('foo', underline=True)

    def test_paren(self, ts, ts_blank):
        if False:
            return 10
        assert str(ts.paren) == '(foo)'
        assert str(ts_blank.paren) == '\x08'

    def test_square(self, ts, ts_blank):
        if False:
            return 10
        assert str(ts.square) == '[foo]'
        assert str(ts_blank.square) == '\x08'

    def test_curly(self, ts, ts_blank):
        if False:
            return 10
        assert str(ts.curly) == '{foo}'
        assert str(ts_blank.curly) == '\x08'

    def test_git(self, ts, ts_blank):
        if False:
            print('Hello World!')
        assert str(ts.git) == ':'.join([style('git', fg='blue'), 'foo'])
        assert str(ts_blank.git) == '\x08'

class TestTTY:

    @pytest.mark.parametrize('color', ['blue', 'red', 'magenta', 'white', 'green', 'black', 'yellow', 'cyan'])
    def test_colors(self, color):
        if False:
            i = 10
            return i + 15
        code = getattr(TTY, color.upper())
        assert code == style('', fg=color, reset=False)

    def test_bold(self):
        if False:
            while True:
                i = 10
        assert TTY.BOLD == style('', bold=True, reset=False)

    def test_blink(self):
        if False:
            print('Hello World!')
        assert TTY.BLINK == style('', blink=True, reset=False)

    def test_underline(self):
        if False:
            for i in range(10):
                print('nop')
        assert TTY.UNDERLINE == style('', underline=True, reset=False)

    def test_dim(self):
        if False:
            for i in range(10):
                print('nop')
        assert TTY.DIM == style('', dim=True, reset=False)