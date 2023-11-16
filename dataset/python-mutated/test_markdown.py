MARKDOWN = 'Heading\n=======\n\nSub-heading\n-----------\n\n### Heading\n\n#### H4 Heading\n\n##### H5 Heading\n\n###### H6 Heading\n\n\nParagraphs are separated\nby a blank line.\n\nTwo spaces at the end of a line  \nproduces a line break.\n\nText attributes _italic_, \n**bold**, `monospace`.\n\nHorizontal rule:\n\n---\n\nBullet list:\n\n  * apples\n  * oranges\n  * pears\n\nNumbered list:\n\n  1. lather\n  2. rinse\n  3. repeat\n\nAn [example](http://example.com).\n\n> Markdown uses email-style > characters for blockquoting.\n>\n> Lorem ipsum\n\n![progress](https://github.com/textualize/rich/raw/master/imgs/progress.gif)\n\n\n```\na=1\n```\n\n```python\nimport this\n```\n\n```somelang\nfoobar\n```\n\n    import this\n\n\n1. List item\n\n       Code block\n'
import io
import re
from rich.console import Console, RenderableType
from rich.markdown import Markdown
re_link_ids = re.compile('id=[\\d\\.\\-]*?;.*?\\x1b')

def replace_link_ids(render: str) -> str:
    if False:
        return 10
    'Link IDs have a random ID and system path which is a problem for\n    reproducible tests.\n\n    '
    return re_link_ids.sub('id=0;foo\x1b', render)

def render(renderable: RenderableType) -> str:
    if False:
        for i in range(10):
            print('nop')
    console = Console(width=100, file=io.StringIO(), color_system='truecolor', legacy_windows=False)
    console.print(renderable)
    output = replace_link_ids(console.file.getvalue())
    print(repr(output))
    return output

def test_markdown_render():
    if False:
        i = 10
        return i + 15
    markdown = Markdown(MARKDOWN)
    rendered_markdown = render(markdown)
    expected = '┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n┃                                             \x1b[1mHeading\x1b[0m                                              ┃\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n\n                                            \x1b[1;4mSub-heading\x1b[0m                                             \n\n                                              \x1b[1mHeading\x1b[0m                                               \n\n                                             \x1b[1;2mH4 Heading\x1b[0m                                             \n\n                                             \x1b[4mH5 Heading\x1b[0m                                             \n\n                                             \x1b[3mH6 Heading\x1b[0m                                             \n\nParagraphs are separated by a blank line.                                                           \n\nTwo spaces at the end of a line                                                                     \nproduces a line break.                                                                              \n\nText attributes \x1b[3mitalic\x1b[0m, \x1b[1mbold\x1b[0m, \x1b[1;36;40mmonospace\x1b[0m.                                                            \n\nHorizontal rule:                                                                                    \n\n\x1b[33m────────────────────────────────────────────────────────────────────────────────────────────────────\x1b[0m\nBullet list:                                                                                        \n\n\x1b[1;33m • \x1b[0mapples                                                                                           \n\x1b[1;33m • \x1b[0moranges                                                                                          \n\x1b[1;33m • \x1b[0mpears                                                                                            \n\nNumbered list:                                                                                      \n\n\x1b[1;33m 1 \x1b[0mlather                                                                                           \n\x1b[1;33m 2 \x1b[0mrinse                                                                                            \n\x1b[1;33m 3 \x1b[0mrepeat                                                                                           \n\nAn \x1b]8;id=0;foo\x1b\\\x1b[4;34mexample\x1b[0m\x1b]8;;\x1b\\.                                                                                         \n\n\x1b[35m▌ \x1b[0m\x1b[35mMarkdown uses email-style > characters for blockquoting.\x1b[0m\x1b[35m                                        \x1b[0m\n\x1b[35m▌ \x1b[0m\x1b[35mLorem ipsum\x1b[0m\x1b[35m                                                                                     \x1b[0m\n\n🌆 \x1b]8;id=0;foo\x1b\\progress\x1b]8;;\x1b\\                                                                                         \n\n\x1b[48;2;39;40;34m                                                                                                    \x1b[0m\n\x1b[48;2;39;40;34m \x1b[0m\x1b[38;2;248;248;242;48;2;39;40;34ma=1\x1b[0m\x1b[48;2;39;40;34m                                                                                               \x1b[0m\x1b[48;2;39;40;34m \x1b[0m\n\x1b[48;2;39;40;34m                                                                                                    \x1b[0m\n\n\x1b[48;2;39;40;34m                                                                                                    \x1b[0m\n\x1b[48;2;39;40;34m \x1b[0m\x1b[38;2;255;70;137;48;2;39;40;34mimport\x1b[0m\x1b[38;2;248;248;242;48;2;39;40;34m \x1b[0m\x1b[38;2;248;248;242;48;2;39;40;34mthis\x1b[0m\x1b[48;2;39;40;34m                                                                                       \x1b[0m\x1b[48;2;39;40;34m \x1b[0m\n\x1b[48;2;39;40;34m                                                                                                    \x1b[0m\n\n\x1b[48;2;39;40;34m                                                                                                    \x1b[0m\n\x1b[48;2;39;40;34m \x1b[0m\x1b[38;2;248;248;242;48;2;39;40;34mfoobar\x1b[0m\x1b[48;2;39;40;34m                                                                                            \x1b[0m\x1b[48;2;39;40;34m \x1b[0m\n\x1b[48;2;39;40;34m                                                                                                    \x1b[0m\n\n\x1b[48;2;39;40;34m                                                                                                    \x1b[0m\n\x1b[48;2;39;40;34m \x1b[0m\x1b[38;2;248;248;242;48;2;39;40;34mimport this\x1b[0m\x1b[48;2;39;40;34m                                                                                       \x1b[0m\x1b[48;2;39;40;34m \x1b[0m\n\x1b[48;2;39;40;34m                                                                                                    \x1b[0m\n\n\x1b[1;33m 1 \x1b[0mList item                                                                                        \n\x1b[1;33m   \x1b[0m\x1b[48;2;39;40;34m                                                                                                 \x1b[0m\n\x1b[1;33m   \x1b[0m\x1b[48;2;39;40;34m \x1b[0m\x1b[38;2;248;248;242;48;2;39;40;34mCode block\x1b[0m\x1b[48;2;39;40;34m                                                                                     \x1b[0m\x1b[48;2;39;40;34m \x1b[0m\n\x1b[1;33m   \x1b[0m\x1b[48;2;39;40;34m                                                                                                 \x1b[0m\n'
    assert rendered_markdown == expected

def test_inline_code():
    if False:
        print('Hello World!')
    markdown = Markdown('inline `import this` code', inline_code_lexer='python', inline_code_theme='emacs')
    result = render(markdown)
    expected = 'inline \x1b[1;38;2;170;34;255;48;2;248;248;248mimport\x1b[0m\x1b[38;2;0;0;0;48;2;248;248;248m \x1b[0m\x1b[1;38;2;0;0;255;48;2;248;248;248mthis\x1b[0m code                                                                             \n'
    print(result)
    print(repr(result))
    assert result == expected

def test_markdown_table():
    if False:
        i = 10
        return i + 15
    markdown = Markdown('| Year |                      Title                       | Director          |  Box Office (USD) |\n|------|:------------------------------------------------:|:------------------|------------------:|\n| 1982 |            *E.T. the Extra-Terrestrial*          | Steven Spielberg  |    $792.9 million |\n| 1980 |  Star Wars: Episode V – The Empire Strikes Back  | Irvin Kershner    |    $538.4 million |\n| 1983 |    Star Wars: Episode VI – Return of the Jedi    | Richard Marquand  |    $475.1 million |\n| 1981 |             Raiders of the Lost Ark              | Steven Spielberg  |    $389.9 million |\n| 1984 |       Indiana Jones and the Temple of Doom       | Steven Spielberg  |    $333.1 million |\n')
    result = render(markdown)
    expected = '\n                                                                                               \n \x1b[1m \x1b[0m\x1b[1mYear\x1b[0m\x1b[1m \x1b[0m \x1b[1m \x1b[0m\x1b[1m                    Title                     \x1b[0m\x1b[1m \x1b[0m \x1b[1m \x1b[0m\x1b[1mDirector        \x1b[0m\x1b[1m \x1b[0m \x1b[1m \x1b[0m\x1b[1mBox Office (USD)\x1b[0m\x1b[1m \x1b[0m \n ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ \n  1982             \x1b[3mE.T. the Extra-Terrestrial\x1b[0m             Steven Spielberg     $792.9 million  \n  1980   Star Wars: Episode V – The Empire Strikes Back   Irvin Kershner       $538.4 million  \n  1983     Star Wars: Episode VI – Return of the Jedi     Richard Marquand     $475.1 million  \n  1981              Raiders of the Lost Ark               Steven Spielberg     $389.9 million  \n  1984        Indiana Jones and the Temple of Doom        Steven Spielberg     $333.1 million  \n                                                                                               \n'
    assert result == expected

def test_inline_styles_in_table():
    if False:
        print('Hello World!')
    'Regression test for https://github.com/Textualize/rich/issues/3115'
    markdown = Markdown('| Year | This **column** displays _the_ movie _title_ ~~description~~ | Director          |  Box Office (USD) |\n|------|:----------------------------------------------------------:|:------------------|------------------:|\n| 1982 | *E.T. the Extra-Terrestrial* ([Wikipedia article](https://en.wikipedia.org/wiki/E.T._the_Extra-Terrestrial)) | Steven Spielberg  |    $792.9 million |\n| 1980 |  Star Wars: Episode V – The *Empire* **Strikes** ~~Back~~  | Irvin Kershner    |    $538.4 million |\n')
    result = render(markdown)
    expected = '\n                                                                                                 \n \x1b[1m \x1b[0m\x1b[1mYear\x1b[0m\x1b[1m \x1b[0m \x1b[1m \x1b[0m\x1b[1mThis \x1b[0m\x1b[1mcolumn\x1b[0m\x1b[1m displays \x1b[0m\x1b[1;3mthe\x1b[0m\x1b[1m movie \x1b[0m\x1b[1;3mtitle\x1b[0m\x1b[1m \x1b[0m\x1b[1;9mdescription\x1b[0m\x1b[1m \x1b[0m \x1b[1m \x1b[0m\x1b[1mDirector        \x1b[0m\x1b[1m \x1b[0m \x1b[1m \x1b[0m\x1b[1mBox Office (USD)\x1b[0m\x1b[1m \x1b[0m \n ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ \n  1982    \x1b[3mE.T. the Extra-Terrestrial\x1b[0m (\x1b]8;id=0;foo\x1b\\\x1b[4;34mWikipedia article\x1b[0m\x1b]8;;\x1b\\)    Steven Spielberg     $792.9 million  \n  1980    Star Wars: Episode V – The \x1b[3mEmpire\x1b[0m \x1b[1mStrikes\x1b[0m \x1b[9mBack\x1b[0m    Irvin Kershner       $538.4 million  \n                                                                                                 \n'
    assert result == expected

def test_inline_styles_with_justification():
    if False:
        i = 10
        return i + 15
    'Regression test for https://github.com/Textualize/rich/issues/3115\n\n    In particular, this tests the interaction between the change that was made to fix\n    #3115 and column text justification.\n    '
    markdown = Markdown('| left | center | right |\n| :- | :-: | -: |\n| This is a long row | because it contains | a fairly long sentence. |\n| a*b* _c_ ~~d~~ e | a*b* _c_ ~~d~~ e | a*b* _c_ ~~d~~ e |')
    result = render(markdown)
    expected = '\n                                                                      \n \x1b[1m \x1b[0m\x1b[1mleft              \x1b[0m\x1b[1m \x1b[0m \x1b[1m \x1b[0m\x1b[1m      center       \x1b[0m\x1b[1m \x1b[0m \x1b[1m \x1b[0m\x1b[1m                  right\x1b[0m\x1b[1m \x1b[0m \n ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ \n  This is a long row   because it contains   a fairly long sentence.  \n  a\x1b[3mb\x1b[0m \x1b[3mc\x1b[0m \x1b[9md\x1b[0m e                  a\x1b[3mb\x1b[0m \x1b[3mc\x1b[0m \x1b[9md\x1b[0m e                        a\x1b[3mb\x1b[0m \x1b[3mc\x1b[0m \x1b[9md\x1b[0m e  \n                                                                      \n'
    assert result == expected

def test_partial_table():
    if False:
        return 10
    markdown = Markdown('| Simple | Table |\n| ------ | ----- ')
    result = render(markdown)
    print(repr(result))
    expected = '\n                  \n \x1b[1m \x1b[0m\x1b[1mSimple\x1b[0m\x1b[1m \x1b[0m \x1b[1m \x1b[0m\x1b[1mTable\x1b[0m\x1b[1m \x1b[0m \n ━━━━━━━━━━━━━━━━ \n                  \n'
    assert result == expected
if __name__ == '__main__':
    markdown = Markdown(MARKDOWN)
    rendered = render(markdown)
    print(rendered)
    print(repr(rendered))