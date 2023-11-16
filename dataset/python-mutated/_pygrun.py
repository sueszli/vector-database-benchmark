__author__ = 'jszheng'
import optparse
import sys
import os
import importlib
from antlr4 import *

def beautify_lisp_string(in_string):
    if False:
        print('Hello World!')
    indent_size = 3
    add_indent = ' ' * indent_size
    out_string = in_string[0]
    indent = ''
    for i in range(1, len(in_string)):
        if in_string[i] == '(' and in_string[i + 1] != ' ':
            indent += add_indent
            out_string += '\n' + indent + '('
        elif in_string[i] == ')':
            out_string += ')'
            if len(indent) > 0:
                indent = indent.replace(add_indent, '', 1)
        else:
            out_string += in_string[i]
    return out_string

def main():
    if False:
        i = 10
        return i + 15
    usage = 'Usage: %prog [options] Grammar_Name Start_Rule'
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-t', '--tree', default=False, action='store_true', help='Print AST tree')
    parser.add_option('-k', '--tokens', dest='token', default=False, action='store_true', help='Show Tokens')
    parser.add_option('-s', '--sll', dest='sll', default=False, action='store_true', help='Show SLL')
    parser.add_option('-d', '--diagnostics', dest='diagnostics', default=False, action='store_true', help='Enable diagnostics error listener')
    parser.add_option('-a', '--trace', dest='trace', default=False, action='store_true', help='Enable Trace')
    (options, remainder) = parser.parse_args()
    if len(remainder) < 2:
        print('ERROR: You have to provide at least 2 arguments!')
        parser.print_help()
        exit(1)
    else:
        grammar = remainder.pop(0)
        start_rule = remainder.pop(0)
        file_list = remainder
    lexerName = grammar + 'Lexer'
    parserName = grammar + 'Parser'
    lexer_file = lexerName + '.py'
    parser_file = parserName + '.py'
    if not os.path.exists(lexer_file):
        print("[ERROR] Can't find lexer file {}!".format(lexer_file))
        print(os.path.realpath('.'))
        exit(1)
    if not os.path.exists(parser_file):
        print("[ERROR] Can't find parser file {}!".format(lexer_file))
        print(os.path.realpath('.'))
        exit(1)
    sys.path.append('.')
    globals().update({'__package__': os.path.basename(os.getcwd())})
    module_lexer = __import__(lexerName, globals(), locals(), lexerName)
    class_lexer = getattr(module_lexer, lexerName)
    module_parser = __import__(parserName, globals(), locals(), parserName)
    class_parser = getattr(module_parser, parserName)

    def process(input_stream, class_lexer, class_parser):
        if False:
            i = 10
            return i + 15
        lexer = class_lexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        token_stream.fill()
        if options.token:
            for tok in token_stream.tokens:
                print(tok)
        if start_rule == 'tokens':
            return
        parser = class_parser(token_stream)
        if options.diagnostics:
            parser.addErrorListener(DiagnosticErrorListener())
            parser._interp.predictionMode = PredictionMode.LL_EXACT_AMBIG_DETECTION
        if options.tree:
            parser.buildParseTrees = True
        if options.sll:
            parser._interp.predictionMode = PredictionMode.SLL
        parser.setTrace(options.trace)
        if hasattr(parser, start_rule):
            func_start_rule = getattr(parser, start_rule)
            parser_ret = func_start_rule()
            if options.tree:
                lisp_tree_str = parser_ret.toStringTree(recog=parser)
                print(beautify_lisp_string(lisp_tree_str))
        else:
            print("[ERROR] Can't find start rule '{}' in parser '{}'".format(start_rule, parserName))
    if len(file_list) == 0:
        input_stream = InputStream(sys.stdin.read())
        process(input_stream, class_lexer, class_parser)
        exit(0)
    for file_name in file_list:
        if os.path.exists(file_name) and os.path.isfile(file_name):
            input_stream = FileStream(file_name)
            process(input_stream, class_lexer, class_parser)
        else:
            print('[ERROR] file {} not exist'.format(os.path.normpath(file_name)))
if __name__ == '__main__':
    main()