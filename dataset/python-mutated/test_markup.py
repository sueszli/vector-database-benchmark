import pytest
from rich.console import Console
from rich.errors import MarkupError
from rich.markup import RE_TAGS, Tag, _parse, escape, render
from rich.text import Span, Text

def test_re_no_match():
    if False:
        while True:
            i = 10
    assert RE_TAGS.match('[True]') == None
    assert RE_TAGS.match('[False]') == None
    assert RE_TAGS.match('[None]') == None
    assert RE_TAGS.match('[1]') == None
    assert RE_TAGS.match('[2]') == None
    assert RE_TAGS.match('[]') == None

def test_re_match():
    if False:
        return 10
    assert RE_TAGS.match('[true]')
    assert RE_TAGS.match('[false]')
    assert RE_TAGS.match('[none]')
    assert RE_TAGS.match('[color(1)]')
    assert RE_TAGS.match('[#ff00ff]')
    assert RE_TAGS.match('[/]')
    assert RE_TAGS.match('[@]')
    assert RE_TAGS.match('[@foo]')
    assert RE_TAGS.match('[@foo=bar]')

def test_escape():
    if False:
        print('Hello World!')
    assert escape('foo[bar]') == 'foo\\[bar]'
    assert escape('foo\\[bar]') == 'foo\\\\\\[bar]'
    assert escape('[5]') == '[5]'
    assert escape('\\[5]') == '\\[5]'
    assert escape('[@foo]') == '\\[@foo]'
    assert escape('[@]') == '\\[@]'
    assert escape('[nil, [nil]]') == '[nil, \\[nil]]'

def test_escape_backslash_end():
    if False:
        while True:
            i = 10
    value = 'C:\\'
    assert escape(value) == 'C:\\\\'
    escaped_tags = f'[red]{escape(value)}[/red]'
    assert escaped_tags == '[red]C:\\\\[/red]'
    escaped_text = Text.from_markup(escaped_tags)
    assert escaped_text.plain == 'C:\\'
    assert escaped_text.spans == [Span(0, 3, 'red')]

def test_render_escape():
    if False:
        while True:
            i = 10
    console = Console(width=80, color_system=None)
    console.begin_capture()
    console.print(escape('[red]'), escape('\\[red]'), escape('\\\\[red]'), escape('\\\\\\[red]'))
    result = console.end_capture()
    expected = '[red] \\[red] \\\\[red] \\\\\\[red]' + '\n'
    assert result == expected

def test_parse():
    if False:
        print('Hello World!')
    result = list(_parse('[foo]hello[/foo][bar]world[/]\\[escaped]'))
    expected = [(0, None, Tag(name='foo', parameters=None)), (10, 'hello', None), (10, None, Tag(name='/foo', parameters=None)), (16, None, Tag(name='bar', parameters=None)), (26, 'world', None), (26, None, Tag(name='/', parameters=None)), (29, '[escaped]', None)]
    print(repr(result))
    assert result == expected

def test_parse_link():
    if False:
        print('Hello World!')
    result = list(_parse('[link=foo]bar[/link]'))
    expected = [(0, None, Tag(name='link', parameters='foo')), (13, 'bar', None), (13, None, Tag(name='/link', parameters=None))]
    assert result == expected

def test_render():
    if False:
        i = 10
        return i + 15
    result = render('[bold]FOO[/bold]')
    assert str(result) == 'FOO'
    assert result.spans == [Span(0, 3, 'bold')]

def test_render_not_tags():
    if False:
        for i in range(10):
            print('nop')
    result = render('[[1], [1,2,3,4], ["hello"], [None], [False], [True]] []')
    assert str(result) == '[[1], [1,2,3,4], ["hello"], [None], [False], [True]] []'
    assert result.spans == []

def test_render_link():
    if False:
        print('Hello World!')
    result = render('[link=foo]FOO[/link]')
    assert str(result) == 'FOO'
    assert result.spans == [Span(0, 3, 'link foo')]

def test_render_combine():
    if False:
        while True:
            i = 10
    result = render('[green]X[blue]Y[/blue]Z[/green]')
    assert str(result) == 'XYZ'
    assert result.spans == [Span(0, 3, 'green'), Span(1, 2, 'blue')]

def test_render_overlap():
    if False:
        for i in range(10):
            print('nop')
    result = render('[green]X[bold]Y[/green]Z[/bold]')
    assert str(result) == 'XYZ'
    assert result.spans == [Span(0, 2, 'green'), Span(1, 3, 'bold')]

def test_adjoint():
    if False:
        while True:
            i = 10
    result = render('[red][blue]B[/blue]R[/red]')
    print(repr(result))
    assert result.spans == [Span(0, 2, 'red'), Span(0, 1, 'blue')]

def test_render_close():
    if False:
        print('Hello World!')
    result = render('[bold]X[/]Y')
    assert str(result) == 'XY'
    assert result.spans == [Span(0, 1, 'bold')]

def test_render_close_ambiguous():
    if False:
        return 10
    result = render('[green]X[bold]Y[/]Z[/]')
    assert str(result) == 'XYZ'
    assert result.spans == [Span(0, 3, 'green'), Span(1, 2, 'bold')]

def test_markup_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(MarkupError):
        assert render('foo[/]')
    with pytest.raises(MarkupError):
        assert render('foo[/bar]')
    with pytest.raises(MarkupError):
        assert render('[foo]hello[/bar]')

def test_markup_escape():
    if False:
        i = 10
        return i + 15
    result = str(render('[dim white][url=[/]'))
    assert result == '[url='

def test_escape_escape():
    if False:
        for i in range(10):
            print('nop')
    result = render('\\\\[bold]FOO')
    assert str(result) == '\\FOO'
    result = render('\\[bold]FOO')
    assert str(result) == '[bold]FOO'
    result = render('\\\\[bold]some text[/]')
    assert str(result) == '\\some text'
    result = render('\\\\\\[bold]some text\\[/]')
    assert str(result) == '\\[bold]some text[/]'
    result = render('\\\\')
    assert str(result) == '\\\\'
    result = render('\\\\\\\\')
    assert str(result) == '\\\\\\\\'

def test_events():
    if False:
        return 10
    result = render("[@click]Hello[/@click] [@click='view.toggle', 'left']World[/]")
    assert str(result) == 'Hello World'

def test_events_broken():
    if False:
        i = 10
        return i + 15
    with pytest.raises(MarkupError):
        render('[@click=sdfwer(sfs)]foo[/]')
    with pytest.raises(MarkupError):
        render("[@click='view.toggle]foo[/]")

def test_render_meta():
    if False:
        print('Hello World!')
    console = Console()
    text = render('foo[@click=close]bar[/]baz')
    assert text.get_style_at_offset(console, 3).meta == {'@click': ('close', ())}
    text = render('foo[@click=close()]bar[/]baz')
    assert text.get_style_at_offset(console, 3).meta == {'@click': ('close', ())}
    text = render("foo[@click=close('dialog')]bar[/]baz")
    assert text.get_style_at_offset(console, 3).meta == {'@click': ('close', ('dialog',))}
    text = render("foo[@click=close('dialog', 3)]bar[/]baz")
    assert text.get_style_at_offset(console, 3).meta == {'@click': ('close', ('dialog', 3))}
    text = render('foo[@click=(1, 2, 3)]bar[/]baz')
    assert text.get_style_at_offset(console, 3).meta == {'@click': (1, 2, 3)}