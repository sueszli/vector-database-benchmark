"""
Demo script for PyMuPDF's `fitz.Story.write_stabilized_with_links()`.

`fitz.Story.write_stabilized_links()` is similar to
`fitz.Story.write_stabilized()` except that it creates a PDF `fitz.Document`
that contains PDF links generated from all internal links in the original html.
"""
import textwrap
import fitz

def rectfn(rect_num, filled):
    if False:
        i = 10
        return i + 15
    '\n    We return one rect per page.\n    '
    rect = fitz.Rect(10, 20, 290, 380)
    mediabox = fitz.Rect(0, 0, 300, 400)
    return (mediabox, rect, None)

def contentfn(positions):
    if False:
        i = 10
        return i + 15
    '\n    Returns html content, with a table of contents derived from `positions`.\n    '
    ret = ''
    ret += textwrap.dedent('\n            <!DOCTYPE html>\n            <body>\n            <h2>Contents</h2>\n            <ul>\n            ')
    for position in positions:
        if position.heading and position.open_close & 1:
            text = position.text if position.text else ''
            if position.id:
                ret += f'    <li><a href="#{position.id}">{text}</a>\n'
            else:
                ret += f'    <li>{text}\n'
            ret += f'        <ul>\n'
            ret += f'        <li>page={position.page_num}\n'
            ret += f'        <li>depth={position.depth}\n'
            ret += f'        <li>heading={position.heading}\n'
            ret += f'        <li>id={position.id!r}\n'
            ret += f'        <li>href={position.href!r}\n'
            ret += f'        <li>rect={position.rect}\n'
            ret += f'        <li>text={text!r}\n'
            ret += f'        <li>open_close={position.open_close}\n'
            ret += f'        </ul>\n'
    ret += '</ul>\n'
    ret += textwrap.dedent(f'\n    \n            <h1>First section</h1>\n            <p>Contents of first section.\n            <ul>\n            <li><a href="#idtest">Link to IDTEST</a>.\n            <li><a href="#nametest">Link to NAMETEST</a>.\n            </ul>\n            \n            <h1>Second section</h1>\n            <p>Contents of second section.\n            <h2>Second section first subsection</h2>\n            \n            <p>Contents of second section first subsection.\n            <p id="idtest">IDTEST\n            \n            <h1>Third section</h1>\n            <p>Contents of third section.\n            <p><a name="nametest">NAMETEST</a>.\n            \n            </body>\n            ')
    ret = ret.strip()
    with open(__file__.replace('.py', '.html'), 'w') as f:
        f.write(ret)
    return ret
out_path = __file__.replace('.py', '.pdf')
document = fitz.Story.write_stabilized_with_links(contentfn, rectfn)
document.save(out_path)