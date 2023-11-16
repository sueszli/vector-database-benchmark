"""Support module for tests for yapf."""
import difflib
import sys
import unittest
from yapf.pytree import blank_line_calculator
from yapf.pytree import comment_splicer
from yapf.pytree import continuation_splicer
from yapf.pytree import pytree_unwrapper
from yapf.pytree import pytree_utils
from yapf.pytree import pytree_visitor
from yapf.pytree import split_penalty
from yapf.pytree import subtype_assigner
from yapf.yapflib import identify_container
from yapf.yapflib import style

class YAPFTest(unittest.TestCase):

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        super(YAPFTest, self).__init__(*args)

    def assertCodeEqual(self, expected_code, code):
        if False:
            return 10
        if code != expected_code:
            msg = ['Code format mismatch:', 'Expected:']
            linelen = style.Get('COLUMN_LIMIT')
            for line in expected_code.splitlines():
                if len(line) > linelen:
                    msg.append('!> %s' % line)
                else:
                    msg.append(' > %s' % line)
            msg.append('Actual:')
            for line in code.splitlines():
                if len(line) > linelen:
                    msg.append('!> %s' % line)
                else:
                    msg.append(' > %s' % line)
            msg.append('Diff:')
            msg.extend(difflib.unified_diff(code.splitlines(), expected_code.splitlines(), fromfile='actual', tofile='expected', lineterm=''))
            self.fail('\n'.join(msg))

def ParseAndUnwrap(code, dumptree=False):
    if False:
        i = 10
        return i + 15
    'Produces logical lines from the given code.\n\n  Parses the code into a tree, performs comment splicing and runs the\n  unwrapper.\n\n  Arguments:\n    code: code to parse as a string\n    dumptree: if True, the parsed pytree (after comment splicing) is dumped\n              to stderr. Useful for debugging.\n\n  Returns:\n    List of logical lines.\n  '
    tree = pytree_utils.ParseCodeToTree(code)
    comment_splicer.SpliceComments(tree)
    continuation_splicer.SpliceContinuations(tree)
    subtype_assigner.AssignSubtypes(tree)
    identify_container.IdentifyContainers(tree)
    split_penalty.ComputeSplitPenalties(tree)
    blank_line_calculator.CalculateBlankLines(tree)
    if dumptree:
        pytree_visitor.DumpPyTree(tree, target_stream=sys.stderr)
    llines = pytree_unwrapper.UnwrapPyTree(tree)
    for lline in llines:
        lline.CalculateFormattingInformation()
    return llines