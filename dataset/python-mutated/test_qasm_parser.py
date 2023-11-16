"""Test for the QASM parser"""
import os
import unittest
import ply
import ddt
from qiskit.qasm import Qasm, QasmError
from qiskit.qasm.node.node import Node
from qiskit.test import QiskitTestCase

def parse(file_path):
    if False:
        while True:
            i = 10
    '\n    Simple helper\n    - file_path: Path to the OpenQASM file\n    - prec: Precision for the returned string\n    '
    qasm = Qasm(file_path)
    return qasm.parse().qasm()

@ddt.ddt
class TestParser(QiskitTestCase):
    """QasmParser"""

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.qasm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qasm')
        self.qasm_file_path = os.path.join(self.qasm_dir, 'example.qasm')
        self.qasm_file_path_fail = os.path.join(self.qasm_dir, 'example_fail.qasm')
        self.qasm_file_path_if = os.path.join(self.qasm_dir, 'example_if.qasm')
        self.qasm_file_path_version_fail = os.path.join(self.qasm_dir, 'example_version_fail.qasm')
        self.qasm_file_path_version_2 = os.path.join(self.qasm_dir, 'example_version_2.qasm')
        self.qasm_file_path_minor_ver_fail = os.path.join(self.qasm_dir, 'example_minor_version_fail.qasm')

    def test_parser(self):
        if False:
            return 10
        'should return a correct response for a valid circuit.'
        res = parse(self.qasm_file_path)
        self.log.info(res)
        starts_expected = 'OPENQASM 2.0;\ngate '
        ends_expected = '\n'.join(['}', 'qreg q[3];', 'qreg r[3];', 'h q;', 'cx q,r;', 'creg c[3];', 'creg d[3];', 'barrier q;', 'measure q -> c;', 'measure r -> d;', ''])
        self.assertEqual(res[:len(starts_expected)], starts_expected)
        self.assertEqual(res[-len(ends_expected):], ends_expected)

    def test_parser_fail(self):
        if False:
            while True:
                i = 10
        'should fail a for a  not valid circuit.'
        self.assertRaisesRegex(QasmError, 'Perhaps there is a missing', parse, file_path=self.qasm_file_path_fail)

    @ddt.data('example_version_fail.qasm', 'example_minor_version_fail.qasm')
    def test_parser_version_fail(self, filename):
        if False:
            print('Hello World!')
        'Ensure versions other than 2.0 or 2 fail.'
        filename = os.path.join(self.qasm_dir, filename)
        with self.assertRaisesRegex(QasmError, "Invalid version: '.+'\\. This module supports OpenQASM 2\\.0 only\\."):
            parse(filename)

    def test_parser_version_2(self):
        if False:
            while True:
                i = 10
        'should succeed for OPENQASM version 2. Parser should automatically add minor verison.'
        res = parse(self.qasm_file_path_version_2)
        version_start = 'OPENQASM 2.0;'
        self.assertEqual(res[:len(version_start)], version_start)

    def test_all_valid_nodes(self):
        if False:
            return 10
        'Test that the tree contains only Node subclasses.'

        def inspect(node):
            if False:
                print('Hello World!')
            'Inspect node children.'
            for child in node.children:
                self.assertTrue(isinstance(child, Node))
                inspect(child)
        qasm = Qasm(self.qasm_file_path)
        res = qasm.parse()
        inspect(res)
        qasm_if = Qasm(self.qasm_file_path_if)
        res_if = qasm_if.parse()
        inspect(res_if)

    def test_generate_tokens(self):
        if False:
            i = 10
            return i + 15
        'Test whether we get only valid tokens.'
        qasm = Qasm(self.qasm_file_path)
        for token in qasm.generate_tokens():
            self.assertTrue(isinstance(token, ply.lex.LexToken))
if __name__ == '__main__':
    unittest.main()