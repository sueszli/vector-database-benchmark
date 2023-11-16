"""
Demo script for PyMuPDF's `fitz.Story.write_stabilized()`.

`fitz.Story.write_stabilized()` is similar to `fitz.Story.write()`,
except instead of taking a fixed html document, it does iterative layout
of dynamically-generated html content (provided by a callback) to a
`fitz.DocumentWriter`.

For example this allows one to add a dynamically-generated table of contents
section while ensuring that page numbers are patched up until stable.
"""
import textwrap
import fitz

def rectfn(rect_num, filled):
    if False:
        print('Hello World!')
    '\n    We return one rect per page.\n    '
    rect = fitz.Rect(10, 20, 290, 380)
    mediabox = fitz.Rect(0, 0, 300, 400)
    return (mediabox, rect, None)

def contentfn(positions):
    if False:
        while True:
            i = 10
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
    ret += textwrap.dedent(f'\n    \n            <h1>First section</h1>\n            <p>Contents of first section.\n            \n            <h1>Second section</h1>\n            <p>Contents of second section.\n            <h2>Second section first subsection</h2>\n            \n            <p>Contents of second section first subsection.\n            \n            <h1>Third section</h1>\n            <p>Contents of third section.\n            \n            </body>\n            ')
    ret = ret.strip()
    with open(__file__.replace('.py', '.html'), 'w') as f:
        f.write(ret)
    return ret
out_path = __file__.replace('.py', '.pdf')
writer = fitz.DocumentWriter(out_path)
fitz.Story.write_stabilized(writer, contentfn, rectfn)
writer.close()