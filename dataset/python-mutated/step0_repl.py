import mal_readline

def READ(str):
    if False:
        for i in range(10):
            print('nop')
    return str

def EVAL(ast, env):
    if False:
        while True:
            i = 10
    return ast

def PRINT(exp):
    if False:
        print('Hello World!')
    return exp

def REP(str):
    if False:
        return 10
    return PRINT(EVAL(READ(str), {}))

def entry_point(argv):
    if False:
        i = 10
        return i + 15
    while True:
        try:
            line = mal_readline.readline('user> ')
            if line == '':
                continue
            print(REP(line))
        except EOFError as e:
            break
        except Exception as e:
            print('Error: %s' % e)
    return 0

def target(*args):
    if False:
        print('Hello World!')
    return entry_point
import sys
if not sys.argv[0].endswith('rpython'):
    entry_point(sys.argv)