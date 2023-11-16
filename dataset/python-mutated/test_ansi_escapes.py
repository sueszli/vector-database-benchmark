from __future__ import absolute_import
import pytest
from behave.formatter import ansi_escapes
from six.moves import range
TEXTS = [u'lorem ipsum', u'Alice and Bob', u'Alice\nBob']
ALL_COLORS = list(ansi_escapes.colors.keys())
CURSOR_UPS = [ansi_escapes.up(count) for count in range(10)]

def colorize(text, color):
    if False:
        for i in range(10):
            print('nop')
    color_escape = ''
    if color:
        color_escape = ansi_escapes.colors[color]
    return color_escape + text + ansi_escapes.escapes['reset']

def colorize_text(text, colors=None):
    if False:
        return 10
    if not colors:
        colors = []
    colors_size = len(colors)
    color_index = 0
    colored_chars = []
    for char in text:
        color = colors[color_index]
        colored_chars.append(colorize(char, color))
        color_index += 1
        if color_index >= colors_size:
            color_index = 0
    return ''.join(colored_chars)

def test_module_setup():
    if False:
        i = 10
        return i + 15
    'Ensure that the module setup (aliases, escapes) occured.'
    aliases_count = len(ansi_escapes.aliases)
    escapes_count = len(ansi_escapes.escapes)
    assert escapes_count >= 2 + aliases_count + aliases_count

class TestStripEscapes(object):

    @pytest.mark.parametrize('text', TEXTS)
    def test_should_return_same_text_without_escapes(self, text):
        if False:
            return 10
        assert text == ansi_escapes.strip_escapes(text)

    @pytest.mark.parametrize('text', ansi_escapes.colors.values())
    def test_should_return_empty_string_for_any_ansi_escape_color(self, text):
        if False:
            print('Hello World!')
        assert '' == ansi_escapes.strip_escapes(text)

    @pytest.mark.parametrize('text', ansi_escapes.escapes.values())
    def test_should_return_empty_string_for_any_ansi_escape(self, text):
        if False:
            while True:
                i = 10
        assert '' == ansi_escapes.strip_escapes(text)

    @pytest.mark.parametrize('text', TEXTS)
    def test_should_strip_color_escapes_from_all_colored_text(self, text):
        if False:
            i = 10
            return i + 15
        colored_text = colorize_text(text, ALL_COLORS)
        assert text == ansi_escapes.strip_escapes(colored_text)
        assert text != colored_text

    @pytest.mark.parametrize('text', TEXTS)
    @pytest.mark.parametrize('color', ALL_COLORS)
    def test_should_strip_color_escapes_from_text(self, text, color):
        if False:
            while True:
                i = 10
        colored_text = colorize(text, color)
        assert text == ansi_escapes.strip_escapes(colored_text)
        assert text != colored_text
        colored_text2 = colorize(text, color) + text
        text2 = text + text
        assert text2 == ansi_escapes.strip_escapes(colored_text2)
        assert text2 != colored_text2

    @pytest.mark.parametrize('text', TEXTS)
    @pytest.mark.parametrize('cursor_up', CURSOR_UPS)
    def test_should_strip_cursor_up_escapes_from_text(self, text, cursor_up):
        if False:
            print('Hello World!')
        colored_text = cursor_up + text + ansi_escapes.escapes['reset']
        assert text == ansi_escapes.strip_escapes(colored_text)
        assert text != colored_text