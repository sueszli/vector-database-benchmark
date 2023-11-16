"""Internal support module for sre"""
MAGIC = 20171005
from _sre import MAXREPEAT, MAXGROUPS

class error(Exception):
    """Exception raised for invalid regular expressions.

    Attributes:

        msg: The unformatted error message
        pattern: The regular expression pattern
        pos: The index in the pattern where compilation failed (may be None)
        lineno: The line corresponding to pos (may be None)
        colno: The column corresponding to pos (may be None)
    """
    __module__ = 're'

    def __init__(self, msg, pattern=None, pos=None):
        if False:
            print('Hello World!')
        self.msg = msg
        self.pattern = pattern
        self.pos = pos
        if pattern is not None and pos is not None:
            msg = '%s at position %d' % (msg, pos)
            if isinstance(pattern, str):
                newline = '\n'
            else:
                newline = b'\n'
            self.lineno = pattern.count(newline, 0, pos) + 1
            self.colno = pos - pattern.rfind(newline, 0, pos)
            if newline in pattern:
                msg = '%s (line %d, column %d)' % (msg, self.lineno, self.colno)
        else:
            self.lineno = self.colno = None
        super().__init__(msg)

class _NamedIntConstant(int):

    def __new__(cls, value, name):
        if False:
            return 10
        self = super(_NamedIntConstant, cls).__new__(cls, value)
        self.name = name
        return self

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.name
    __reduce__ = None
MAXREPEAT = _NamedIntConstant(MAXREPEAT, 'MAXREPEAT')

def _makecodes(names):
    if False:
        return 10
    names = names.strip().split()
    items = [_NamedIntConstant(i, name) for (i, name) in enumerate(names)]
    globals().update({item.name: item for item in items})
    return items
OPCODES = _makecodes('\n    FAILURE SUCCESS\n\n    ANY ANY_ALL\n    ASSERT ASSERT_NOT\n    AT\n    BRANCH\n    CALL\n    CATEGORY\n    CHARSET BIGCHARSET\n    GROUPREF GROUPREF_EXISTS\n    IN\n    INFO\n    JUMP\n    LITERAL\n    MARK\n    MAX_UNTIL\n    MIN_UNTIL\n    NOT_LITERAL\n    NEGATE\n    RANGE\n    REPEAT\n    REPEAT_ONE\n    SUBPATTERN\n    MIN_REPEAT_ONE\n\n    GROUPREF_IGNORE\n    IN_IGNORE\n    LITERAL_IGNORE\n    NOT_LITERAL_IGNORE\n\n    GROUPREF_LOC_IGNORE\n    IN_LOC_IGNORE\n    LITERAL_LOC_IGNORE\n    NOT_LITERAL_LOC_IGNORE\n\n    GROUPREF_UNI_IGNORE\n    IN_UNI_IGNORE\n    LITERAL_UNI_IGNORE\n    NOT_LITERAL_UNI_IGNORE\n    RANGE_UNI_IGNORE\n\n    MIN_REPEAT MAX_REPEAT\n')
del OPCODES[-2:]
ATCODES = _makecodes('\n    AT_BEGINNING AT_BEGINNING_LINE AT_BEGINNING_STRING\n    AT_BOUNDARY AT_NON_BOUNDARY\n    AT_END AT_END_LINE AT_END_STRING\n\n    AT_LOC_BOUNDARY AT_LOC_NON_BOUNDARY\n\n    AT_UNI_BOUNDARY AT_UNI_NON_BOUNDARY\n')
CHCODES = _makecodes('\n    CATEGORY_DIGIT CATEGORY_NOT_DIGIT\n    CATEGORY_SPACE CATEGORY_NOT_SPACE\n    CATEGORY_WORD CATEGORY_NOT_WORD\n    CATEGORY_LINEBREAK CATEGORY_NOT_LINEBREAK\n\n    CATEGORY_LOC_WORD CATEGORY_LOC_NOT_WORD\n\n    CATEGORY_UNI_DIGIT CATEGORY_UNI_NOT_DIGIT\n    CATEGORY_UNI_SPACE CATEGORY_UNI_NOT_SPACE\n    CATEGORY_UNI_WORD CATEGORY_UNI_NOT_WORD\n    CATEGORY_UNI_LINEBREAK CATEGORY_UNI_NOT_LINEBREAK\n')
OP_IGNORE = {LITERAL: LITERAL_IGNORE, NOT_LITERAL: NOT_LITERAL_IGNORE}
OP_LOCALE_IGNORE = {LITERAL: LITERAL_LOC_IGNORE, NOT_LITERAL: NOT_LITERAL_LOC_IGNORE}
OP_UNICODE_IGNORE = {LITERAL: LITERAL_UNI_IGNORE, NOT_LITERAL: NOT_LITERAL_UNI_IGNORE}
AT_MULTILINE = {AT_BEGINNING: AT_BEGINNING_LINE, AT_END: AT_END_LINE}
AT_LOCALE = {AT_BOUNDARY: AT_LOC_BOUNDARY, AT_NON_BOUNDARY: AT_LOC_NON_BOUNDARY}
AT_UNICODE = {AT_BOUNDARY: AT_UNI_BOUNDARY, AT_NON_BOUNDARY: AT_UNI_NON_BOUNDARY}
CH_LOCALE = {CATEGORY_DIGIT: CATEGORY_DIGIT, CATEGORY_NOT_DIGIT: CATEGORY_NOT_DIGIT, CATEGORY_SPACE: CATEGORY_SPACE, CATEGORY_NOT_SPACE: CATEGORY_NOT_SPACE, CATEGORY_WORD: CATEGORY_LOC_WORD, CATEGORY_NOT_WORD: CATEGORY_LOC_NOT_WORD, CATEGORY_LINEBREAK: CATEGORY_LINEBREAK, CATEGORY_NOT_LINEBREAK: CATEGORY_NOT_LINEBREAK}
CH_UNICODE = {CATEGORY_DIGIT: CATEGORY_UNI_DIGIT, CATEGORY_NOT_DIGIT: CATEGORY_UNI_NOT_DIGIT, CATEGORY_SPACE: CATEGORY_UNI_SPACE, CATEGORY_NOT_SPACE: CATEGORY_UNI_NOT_SPACE, CATEGORY_WORD: CATEGORY_UNI_WORD, CATEGORY_NOT_WORD: CATEGORY_UNI_NOT_WORD, CATEGORY_LINEBREAK: CATEGORY_UNI_LINEBREAK, CATEGORY_NOT_LINEBREAK: CATEGORY_UNI_NOT_LINEBREAK}
SRE_FLAG_TEMPLATE = 1
SRE_FLAG_IGNORECASE = 2
SRE_FLAG_LOCALE = 4
SRE_FLAG_MULTILINE = 8
SRE_FLAG_DOTALL = 16
SRE_FLAG_UNICODE = 32
SRE_FLAG_VERBOSE = 64
SRE_FLAG_DEBUG = 128
SRE_FLAG_ASCII = 256
SRE_INFO_PREFIX = 1
SRE_INFO_LITERAL = 2
SRE_INFO_CHARSET = 4
if __name__ == '__main__':

    def dump(f, d, prefix):
        if False:
            i = 10
            return i + 15
        items = sorted(d)
        for item in items:
            f.write('#define %s_%s %d\n' % (prefix, item, item))
    with open('sre_constants.h', 'w') as f:
        f.write("/*\n * Secret Labs' Regular Expression Engine\n *\n * regular expression matching engine\n *\n * NOTE: This file is generated by sre_constants.py.  If you need\n * to change anything in here, edit sre_constants.py and run it.\n *\n * Copyright (c) 1997-2001 by Secret Labs AB.  All rights reserved.\n *\n * See the _sre.c file for information on usage and redistribution.\n */\n\n")
        f.write('#define SRE_MAGIC %d\n' % MAGIC)
        dump(f, OPCODES, 'SRE_OP')
        dump(f, ATCODES, 'SRE')
        dump(f, CHCODES, 'SRE')
        f.write('#define SRE_FLAG_TEMPLATE %d\n' % SRE_FLAG_TEMPLATE)
        f.write('#define SRE_FLAG_IGNORECASE %d\n' % SRE_FLAG_IGNORECASE)
        f.write('#define SRE_FLAG_LOCALE %d\n' % SRE_FLAG_LOCALE)
        f.write('#define SRE_FLAG_MULTILINE %d\n' % SRE_FLAG_MULTILINE)
        f.write('#define SRE_FLAG_DOTALL %d\n' % SRE_FLAG_DOTALL)
        f.write('#define SRE_FLAG_UNICODE %d\n' % SRE_FLAG_UNICODE)
        f.write('#define SRE_FLAG_VERBOSE %d\n' % SRE_FLAG_VERBOSE)
        f.write('#define SRE_FLAG_DEBUG %d\n' % SRE_FLAG_DEBUG)
        f.write('#define SRE_FLAG_ASCII %d\n' % SRE_FLAG_ASCII)
        f.write('#define SRE_INFO_PREFIX %d\n' % SRE_INFO_PREFIX)
        f.write('#define SRE_INFO_LITERAL %d\n' % SRE_INFO_LITERAL)
        f.write('#define SRE_INFO_CHARSET %d\n' % SRE_INFO_CHARSET)
    print('done')