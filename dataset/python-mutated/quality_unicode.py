import re
import fnmatch
message_unicode_B = 'File contains a unicode character : %s, line %s. But not in the whitelist. Add the file to the whitelist in ' + __file__
message_unicode_D = 'File does not contain a unicode character : %s.but is in the whitelist. Remove the file from the whitelist in ' + __file__
encoding_header_re = re.compile('^[ \\t\\f]*#.*?coding[:=][ \\t]*([-_.a-zA-Z0-9]+)')
unicode_whitelist = ['*/bin/authors_update.py', '*/bin/mailmap_check.py', '*/sympy/testing/tests/test_code_quality.py', '*/sympy/physics/vector/tests/test_printing.py', '*/physics/quantum/tests/test_printing.py', '*/sympy/vector/tests/test_printing.py', '*/sympy/parsing/tests/test_sympy_parser.py', '*/sympy/printing/pretty/stringpict.py', '*/sympy/printing/pretty/tests/test_pretty.py', '*/sympy/printing/tests/test_conventions.py', '*/sympy/printing/tests/test_preview.py', '*/liealgebras/type_g.py', '*/liealgebras/weyl_group.py', '*/liealgebras/tests/test_type_G.py', '*/sympy/physics/wigner.py', '*/sympy/physics/optics/polarization.py', '*/sympy/physics/mechanics/joint.py', '*/sympy/polys/matrices/domainmatrix.py', '*/sympy/matrices/repmatrix.py']
unicode_strict_whitelist = ['*/sympy/parsing/latex/_antlr/__init__.py', '*/sympy/parsing/tests/test_mathematica.py']

def _test_this_file_encoding(fname, test_file, unicode_whitelist=unicode_whitelist, unicode_strict_whitelist=unicode_strict_whitelist):
    if False:
        return 10
    'Test helper function for unicode test\n\n    The test may have to operate on filewise manner, so it had moved\n    to a separate process.\n    '
    has_unicode = False
    is_in_whitelist = False
    is_in_strict_whitelist = False
    for patt in unicode_whitelist:
        if fnmatch.fnmatch(fname, patt):
            is_in_whitelist = True
            break
    for patt in unicode_strict_whitelist:
        if fnmatch.fnmatch(fname, patt):
            is_in_strict_whitelist = True
            is_in_whitelist = True
            break
    if is_in_whitelist:
        for (idx, line) in enumerate(test_file):
            try:
                line.encode(encoding='ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                has_unicode = True
        if not has_unicode and (not is_in_strict_whitelist):
            assert False, message_unicode_D % fname
    else:
        for (idx, line) in enumerate(test_file):
            try:
                line.encode(encoding='ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                assert False, message_unicode_B % (fname, idx + 1)