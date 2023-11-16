import re
from epydoc import docstringparser as dsp
CYTHON_SIGNATURE_RE = re.compile('^\\s*((?P<class>\\w+)\\.)?' + '(?P<func>\\w+)' + '\\(((?P<self>(?:self|cls|mcs)),?)?(?P<params>.*)\\)' + '(\\s*(->)\\s*(?P<return>\\w+(?:\\s*\\w+)))?' + '\\s*(?:\\n|$)')
parse_signature = dsp.parse_function_signature

def parse_function_signature(func_doc, doc_source, docformat, parse_errors):
    if False:
        i = 10
        return i + 15
    PYTHON_SIGNATURE_RE = dsp._SIGNATURE_RE
    assert PYTHON_SIGNATURE_RE is not CYTHON_SIGNATURE_RE
    try:
        dsp._SIGNATURE_RE = CYTHON_SIGNATURE_RE
        found = parse_signature(func_doc, doc_source, docformat, parse_errors)
        dsp._SIGNATURE_RE = PYTHON_SIGNATURE_RE
        if not found:
            found = parse_signature(func_doc, doc_source, docformat, parse_errors)
        return found
    finally:
        dsp._SIGNATURE_RE = PYTHON_SIGNATURE_RE
dsp.parse_function_signature = parse_function_signature
from epydoc.cli import cli
cli()