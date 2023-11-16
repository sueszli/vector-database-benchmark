import mal_readline
import mal_types as types
import reader, printer

def READ(str):
    if False:
        i = 10
        return i + 15
    return reader.read_str(str)

def EVAL(ast, env):
    if False:
        print('Hello World!')
    return ast

def PRINT(exp):
    if False:
        i = 10
        return i + 15
    return printer._pr_str(exp)

def REP(str):
    if False:
        for i in range(10):
            print('nop')
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
        except reader.Blank:
            continue
        except types.MalException as e:
            print(u'Error: %s' % printer._pr_str(e.object, False))
        except Exception as e:
            print('Error: %s' % e)
    return 0

def target(*args):
    if False:
        return 10
    return entry_point
import sys
if not sys.argv[0].endswith('rpython'):
    entry_point(sys.argv)