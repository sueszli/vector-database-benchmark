"""Test pyc/generate_opcode_diffs.py."""
import json
import subprocess
import textwrap
import types
from unittest import mock
from pytype.pyc import generate_opcode_diffs
import unittest

class GenerateOpcodeDiffsTest(unittest.TestCase):

    def _generate_diffs(self):
        if False:
            return 10
        with mock.patch.object(subprocess, 'run') as mock_run:
            mapping_38 = json.dumps({'opmap': {'DO_THIS': 1, 'I_MOVE': 2, 'DO_EIGHT': 5, 'JUMP': 8}, 'opname': ['<0>', 'DO_THIS', 'I_MOVE', '<3>', '<4>', 'DO_EIGHT', '<6>', '<7>', 'JUMP'], 'HAVE_ARGUMENT': 3, 'HAS_CONST': [], 'HAS_NAME': [], 'HAS_JREL': []})
            mapping_39 = json.dumps({'opmap': {'I_MOVE': 3, 'DO_THAT': 4, 'DO_THAT_TOO': 5, 'DO_NINE': 7, 'JUMP': 8}, 'opname': ['<0>', '<1>', '<2>', 'I_MOVE', 'DO_THAT', 'DO_THAT_TOO', '<6>', 'DO_NINE', 'JUMP'], 'HAVE_ARGUMENT': 6, 'HAS_CONST': [7], 'HAS_NAME': [5, 7], 'HAS_JREL': [8]})
            mock_run.side_effect = [types.SimpleNamespace(stdout=mapping_38), types.SimpleNamespace(stdout=mapping_39)]
            return generate_opcode_diffs.generate_diffs(['3.8', '3.9'])

    def test_classes(self):
        if False:
            print('Hello World!')
        (classes, _, _, _) = self._generate_diffs()
        (i_move, do_that, do_that_too, do_nine, jump) = classes
        self.assertMultiLineEqual('\n'.join(i_move), textwrap.dedent('\n      class I_MOVE(Opcode):\n        __slots__ = ()\n    ').strip())
        self.assertMultiLineEqual('\n'.join(do_that), textwrap.dedent('\n      class DO_THAT(Opcode):\n        __slots__ = ()\n    ').strip())
        self.assertMultiLineEqual('\n'.join(do_that_too), textwrap.dedent('\n      class DO_THAT_TOO(Opcode):\n        FLAGS = HAS_NAME\n        __slots__ = ()\n    ').strip())
        self.assertMultiLineEqual('\n'.join(do_nine), textwrap.dedent('\n      class DO_NINE(OpcodeWithArg):\n        FLAGS = HAS_ARGUMENT | HAS_CONST | HAS_NAME\n        __slots__ = ()\n    ').strip())
        self.assertMultiLineEqual('\n'.join(jump), textwrap.dedent('\n      class JUMP(OpcodeWithArg):\n        FLAGS = HAS_ARGUMENT | HAS_JREL\n        __slots__ = ()\n    ').strip())

    def test_diff(self):
        if False:
            i = 10
            return i + 15
        (_, diff, _, _) = self._generate_diffs()
        self.assertMultiLineEqual('\n'.join(diff), textwrap.dedent('\n      1: None,  # was DO_THIS in 3.8\n      2: None,  # was I_MOVE in 3.8\n      3: I_MOVE,\n      4: DO_THAT,\n      5: DO_THAT_TOO,  # was DO_EIGHT in 3.8\n      7: DO_NINE,\n    ').strip())

    def test_stubs(self):
        if False:
            print('Hello World!')
        (_, _, stubs, _) = self._generate_diffs()
        (do_that, do_that_too, do_nine) = stubs
        self.assertMultiLineEqual('\n'.join(do_that), textwrap.dedent('\n      def byte_DO_THAT(self, state, op):\n        del op\n        return state\n    ').strip())
        self.assertMultiLineEqual('\n'.join(do_that_too), textwrap.dedent('\n      def byte_DO_THAT_TOO(self, state, op):\n        del op\n        return state\n    ').strip())
        self.assertMultiLineEqual('\n'.join(do_nine), textwrap.dedent('\n      def byte_DO_NINE(self, state, op):\n        del op\n        return state\n    ').strip())

    def test_impl_changed(self):
        if False:
            return 10
        (_, _, _, impl_changed) = self._generate_diffs()
        self.assertEqual(impl_changed, ['I_MOVE', 'JUMP'])
if __name__ == '__main__':
    unittest.main()