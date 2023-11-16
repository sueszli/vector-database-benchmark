import sys
import getopt
defs = []
undefs = []

def main():
    if False:
        for i in range(10):
            print('nop')
    (opts, args) = getopt.getopt(sys.argv[1:], 'D:U:')
    for (o, a) in opts:
        if o == '-D':
            defs.append(a)
        if o == '-U':
            undefs.append(a)
    if not args:
        args = ['-']
    for filename in args:
        if filename == '-':
            process(sys.stdin, sys.stdout)
        else:
            with open(filename) as f:
                process(f, sys.stdout)

def process(fpi, fpo):
    if False:
        return 10
    keywords = ('if', 'ifdef', 'ifndef', 'else', 'endif')
    ok = 1
    stack = []
    while 1:
        line = fpi.readline()
        if not line:
            break
        while line[-2:] == '\\\n':
            nextline = fpi.readline()
            if not nextline:
                break
            line = line + nextline
        tmp = line.strip()
        if tmp[:1] != '#':
            if ok:
                fpo.write(line)
            continue
        tmp = tmp[1:].strip()
        words = tmp.split()
        keyword = words[0]
        if keyword not in keywords:
            if ok:
                fpo.write(line)
            continue
        if keyword in ('ifdef', 'ifndef') and len(words) == 2:
            if keyword == 'ifdef':
                ko = 1
            else:
                ko = 0
            word = words[1]
            if word in defs:
                stack.append((ok, ko, word))
                if not ko:
                    ok = 0
            elif word in undefs:
                stack.append((ok, not ko, word))
                if ko:
                    ok = 0
            else:
                stack.append((ok, -1, word))
                if ok:
                    fpo.write(line)
        elif keyword == 'if':
            stack.append((ok, -1, ''))
            if ok:
                fpo.write(line)
        elif keyword == 'else' and stack:
            (s_ok, s_ko, s_word) = stack[-1]
            if s_ko < 0:
                if ok:
                    fpo.write(line)
            else:
                s_ko = not s_ko
                ok = s_ok
                if not s_ko:
                    ok = 0
                stack[-1] = (s_ok, s_ko, s_word)
        elif keyword == 'endif' and stack:
            (s_ok, s_ko, s_word) = stack[-1]
            if s_ko < 0:
                if ok:
                    fpo.write(line)
            del stack[-1]
            ok = s_ok
        else:
            sys.stderr.write('Unknown keyword %s\n' % keyword)
    if stack:
        sys.stderr.write('stack: %s\n' % stack)
if __name__ == '__main__':
    main()