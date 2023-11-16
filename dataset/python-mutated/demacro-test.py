import unittest
import re
from pix2tex.dataset.demacro import pydemacro

def norm(s):
    if False:
        for i in range(10):
            print('nop')
    s = re.sub('\\n+', '\n', s)
    s = re.sub('\\s+', ' ', s)
    return s.strip()

def f(s):
    if False:
        while True:
            i = 10
    return norm(pydemacro(s))

class TestDemacroCases(unittest.TestCase):

    def test_noargs(self):
        if False:
            for i in range(10):
                print('nop')
        inp = '\n        \\newcommand*{\\noargs}{sample text}\n        \\noargs[a]\\noargs{b}\\noargs\n        '
        expected = 'sample text[a]sample text{b}sample text'
        self.assertEqual(f(inp), norm(expected))

    def test_optional_arg(self):
        if False:
            while True:
                i = 10
        inp = '\n        \\newcommand{\\example}[2][YYY]{Mandatory arg: #2; Optional arg: #1.}     \n        \\example{BBB}\n        \\example[XXX]{AAA}\n        '
        expected = '\n        Mandatory arg: BBB; Optional arg: YYY.\n        Mandatory arg: AAA; Optional arg: XXX.\n        '
        self.assertEqual(f(inp), norm(expected))

    def test_optional_arg_and_positional_args(self):
        if False:
            print('Hello World!')
        inp = '\n        \\newcommand{\\plusbinomial}[3][2]{(#2 + #3)^{#1}}\n        \\plusbinomial[4]{y}{x}\n        '
        expected = '(y + x)^{4}'
        self.assertEqual(f(inp), norm(expected))

    def test_alt_definition1(self):
        if False:
            while True:
                i = 10
        inp = '\n        \\newcommand\\d{replacement}\n        \\d\n        '
        expected = 'replacement'
        self.assertEqual(f(inp), norm(expected))

    def test_arg_with_bs_and_cb(self):
        if False:
            i = 10
            return i + 15
        inp = '\n        \\newcommand{\\eq}[1]{\\begin{equation}#1\\end{equation}}\n        \\eq{\\sqrt{2}\\approx1.4}\n        \\eq[unexpected argument]{\\sqrt{2}\\approx1.4}\n        '
        expected = '\n        \\begin{equation}\\sqrt{2}\\approx1.4\\end{equation}\n        \\begin{equation}\\sqrt{2}\\approx1.4\\end{equation}\n        '
        self.assertEqual(f(inp), norm(expected))

    def test_multiline_definition(self):
        if False:
            for i in range(10):
                print('nop')
        inp = '\n        \\newcommand{\\multiline}[2]{%\n        Arg 1: \\bf{#1}\n        Arg 2: #2\n        }\n        \\multiline{1}{two}\n        '
        expected = '\n        Arg 1: \\bf{1}\n        Arg 2: two\n        '
        self.assertEqual(f(inp), norm(expected))

    def test_multiline_definition_alt1(self):
        if False:
            for i in range(10):
                print('nop')
        inp = '\n        \\newcommand{\\identity}[1]\n        {#1}\n        \\identity{x}\n        '
        expected = 'x'
        self.assertEqual(f(inp), norm(expected))

    def test_multiline_definition_alt2(self):
        if False:
            print('Hello World!')
        inp = '\n        \\newcommand\n        {\\identity}[1]{#1}\n        \\identity{x}\n        '
        expected = 'x'
        self.assertEqual(f(inp), norm(expected))

    def test_multiline_definition_alt3(self):
        if False:
            for i in range(10):
                print('nop')
        inp = '\n        \\newcommand\n        {\\identity}[1]\n        {#1}\n        \\identity{x}\n        '
        expected = 'x'
        self.assertEqual(f(inp), norm(expected))

    def test_multiline_definition_alt4(self):
        if False:
            return 10
        inp = '\n        \\newcommand\n        {\\identity}\n        [1]\n        {#1}\n        \\identity{x}\n        '
        expected = 'x'
        self.assertEqual(f(inp), norm(expected))

    def test_nested_definition(self):
        if False:
            for i in range(10):
                print('nop')
        inp = '\n        \\newcommand{\\cmd}[1]{command #1}\n        \\newcommand{\\nested}[2]{\\cmd{#1} \\cmd{#2}}\n        \\nested{\\alpha}{\\beta}\n        '
        expected = '\n        command \\alpha command \\beta\n        '
        self.assertEqual(f(inp), norm(expected))

    def test_def(self):
        if False:
            print('Hello World!')
        inp = '\n        \\def\\defcheck#1#2{Defcheck arg1: #1 arg2: #2}\n        \\defcheck{1}{two}\n        '
        expected = '\n        Defcheck arg1: 1 arg2: two\n        '
        self.assertEqual(f(inp), norm(expected))

    def test_multi_def_lines_alt0(self):
        if False:
            print('Hello World!')
        inp = '\\def\\be{\\begin{equation}} \\def\\ee{\\end{equation}} %some comment\n        \\be\n        1+1=2\n        \\ee'
        expected = '\n        \\begin{equation}\n        1+1=2\n        \\end{equation}\n        '
        self.assertEqual(f(inp), norm(expected))

    def test_multi_def_lines_alt1(self):
        if False:
            i = 10
            return i + 15
        inp = '\\def\\be{\\begin{equation}}\\def\\ee{\\end{equation}}\n        \\be\n        1+1=2\n        \\ee'
        expected = '\n        \\begin{equation}\n        1+1=2\n        \\end{equation}\n        '
        self.assertEqual(f(inp), norm(expected))

    def test_multi_def_lines_alt2(self):
        if False:
            i = 10
            return i + 15
        inp = '\\def\n        \\be{\\begin{equation}}\n        \\def\\ee\n        {\\end{equation}}\n        \\be\n        1+1=2\n        \\ee'
        expected = '\n        \\begin{equation}\n        1+1=2\n        \\end{equation}\n        '
        self.assertEqual(f(inp), norm(expected))

    def test_multi_def_lines_alt3(self):
        if False:
            i = 10
            return i + 15
        inp = '\n        \\def\\be\n        {\n            \\begin{equation}\n        }\n        \\def\n        \\ee\n        {\\end{equation}}\n        \\be\n        1+1=2\n        \\ee'
        expected = '\n        \\begin{equation}\n        1+1=2\n        \\end{equation}\n        '
        self.assertEqual(f(inp), norm(expected))

    def test_let_alt0(self):
        if False:
            return 10
        inp = '\\let\\a\\alpha\\let\\b=\\beta\n        \\a \\b'
        expected = '\\alpha \\beta'
        self.assertEqual(f(inp), norm(expected))

    def test_let_alt1(self):
        if False:
            for i in range(10):
                print('nop')
        inp = '\\let\\a\\alpha \\let\\b=\\beta\n        \\a \\b'
        expected = '\\alpha \\beta'
        self.assertEqual(f(inp), norm(expected))

    def test_let_alt2(self):
        if False:
            return 10
        inp = '\\let\\a\\alpha \\let\\b=\\beta\n        \\a \\b'
        expected = '\\alpha \\beta'
        self.assertEqual(f(inp), norm(expected))

    def test_let_alt3(self):
        if False:
            print('Hello World!')
        inp = '\n        \\let\n        \\a\n        \\alpha\n        \\let\\b=\n        \\beta\n        \\a \\b'
        expected = '\\alpha \\beta'
        self.assertEqual(f(inp), norm(expected))
if __name__ == '__main__':
    unittest.main()