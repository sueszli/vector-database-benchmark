"""
Check approx. equality of search quads versus quads recovered from
text extractions.
"""
import os
import fitz
scriptdir = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(scriptdir, 'resources', 'quad-calc-0.pdf')

def test_quadcalc():
    if False:
        for i in range(10):
            print('nop')
    text = ' angle 327'
    doc = fitz.open(filename)
    page = doc[0]
    block = page.get_text('dict', flags=0)['blocks'][0]
    line = block['lines'][0]
    lineq = fitz.recover_line_quad(line, spans=line['spans'][-1:])
    rl = page.search_for(text, quads=True)
    searchq = rl[0]
    assert abs(searchq.ul - lineq.ul) <= 0.0001
    assert abs(searchq.ur - lineq.ur) <= 0.0001
    assert abs(searchq.ll - lineq.ll) <= 0.0001
    assert abs(searchq.lr - lineq.lr) <= 0.0001