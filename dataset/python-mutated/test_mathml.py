import os
from textwrap import dedent
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.mathml import apply_xsl
lxml = import_module('lxml')
path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_xxe.py'))

def test_xxe():
    if False:
        while True:
            i = 10
    assert os.path.isfile(path)
    if not lxml:
        skip('lxml not installed.')
    mml = dedent(f'\n        <!--?xml version="1.0" ?-->\n        <!DOCTYPE replace [<!ENTITY ent SYSTEM "file://{path}"> ]>\n        <userInfo>\n        <firstName>John</firstName>\n        <lastName>&ent;</lastName>\n        </userInfo>\n        ')
    xsl = 'mathml/data/simple_mmlctop.xsl'
    res = apply_xsl(mml, xsl)
    assert res == '<?xml version="1.0"?>\n<userInfo>\n<firstName>John</firstName>\n<lastName/>\n</userInfo>\n'