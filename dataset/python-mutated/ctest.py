import sys
sys.setrecursionlimit(4000)
import antlr4
from parser.cparser import CParser
from parser.clexer import CLexer
from datetime import datetime
import cProfile

class ErrorListener(antlr4.error.ErrorListener.ErrorListener):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(ErrorListener, self).__init__()
        self.errored_out = False

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        if False:
            return 10
        self.errored_out = True

def sub():
    if False:
        i = 10
        return i + 15
    input_stream = antlr4.FileStream('c.c')
    lexer = CLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = CParser(token_stream)
    errors = ErrorListener()
    parser.addErrorListener(errors)
    tree = parser.compilationUnit()

def main():
    if False:
        print('Hello World!')
    before = datetime.now()
    sub()
    after = datetime.now()
    print(str(after - before))
if __name__ == '__main__':
    cProfile.run('main()', sort='tottime')