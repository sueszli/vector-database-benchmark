import antlr4
from antlr4 import InputStream, CommonTokenStream, TerminalNode
from antlr4.xpath.XPath import XPath
import unittest
from expr.ExprParser import ExprParser
from expr.ExprLexer import ExprLexer

def tokenToString(token, ruleNames):
    if False:
        while True:
            i = 10
    if isinstance(token, TerminalNode):
        return str(token)
    else:
        return ruleNames[token.getRuleIndex()]

class XPathTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.input_stream = InputStream('def f(x,y) { x = 3+4; y; ; }\ndef g(x) { return 1+2*x; }\n')
        self.lexer = ExprLexer(self.input_stream)
        self.stream = CommonTokenStream(self.lexer)
        self.stream.fill()
        self.parser = ExprParser(self.stream)
        self.tree = self.parser.prog()

    def testValidPaths(self):
        if False:
            i = 10
            return i + 15
        valid_paths = ['/prog/func', '/prog/*', '/*/func', 'prog', '/prog', '/*', '*', '//ID', '//expr/primary/ID', '//body//ID', "//'return'", '//RETURN', '//primary/*', '//func/*/stat', "/prog/func/'def'", "//stat/';'", '//expr/primary/!ID', '//expr/!primary', '//!*', '/!*', '//expr//ID']
        expected_results = ['[func, func]', '[func, func]', '[func, func]', '[prog]', '[prog]', '[prog]', '[prog]', '[f, x, y, x, y, g, x, x]', '[y, x]', '[x, y, x]', '[return]', '[return]', '[3, 4, y, 1, 2, x]', '[stat, stat, stat, stat]', '[def, def]', '[;, ;, ;, ;]', '[3, 4, 1, 2]', '[expr, expr, expr, expr, expr, expr]', '[]', '[]', '[y, x]']
        for (path, expected) in zip(valid_paths, expected_results):
            res = XPath.findAll(self.tree, path, self.parser)
            res_str = ', '.join([tokenToString(token, self.parser.ruleNames) for token in res])
            res_str = '[%s]' % res_str
            self.assertEqual(res_str, expected, 'Failed test %s' % path)
if __name__ == '__main__':
    unittest.main()