import re
makevardef = re.compile('^([a-zA-Z0-9_]+)[ \t]*=(.*)')

def getmakevars(filename):
    if False:
        return 10
    variables = {}
    fp = open(filename)
    pendingline = ''
    try:
        while 1:
            line = fp.readline()
            if pendingline:
                line = pendingline + line
                pendingline = ''
            if not line:
                break
            if line.endswith('\\\n'):
                pendingline = line[:-2]
            matchobj = makevardef.match(line)
            if not matchobj:
                continue
            (name, value) = matchobj.group(1, 2)
            i = value.find('#')
            if i >= 0:
                value = value[:i]
            value = value.strip()
            variables[name] = value
    finally:
        fp.close()
    return variables
setupvardef = re.compile('^([a-zA-Z0-9_]+)=(.*)')

def getsetupinfo(filename):
    if False:
        while True:
            i = 10
    modules = {}
    variables = {}
    fp = open(filename)
    pendingline = ''
    try:
        while 1:
            line = fp.readline()
            if pendingline:
                line = pendingline + line
                pendingline = ''
            if not line:
                break
            i = line.find('#')
            if i >= 0:
                line = line[:i]
            if line.endswith('\\\n'):
                pendingline = line[:-2]
                continue
            matchobj = setupvardef.match(line)
            if matchobj:
                (name, value) = matchobj.group(1, 2)
                variables[name] = value.strip()
            else:
                words = line.split()
                if words:
                    modules[words[0]] = words[1:]
    finally:
        fp.close()
    return (modules, variables)

def test():
    if False:
        return 10
    import sys
    import os
    if not sys.argv[1:]:
        print('usage: python parsesetup.py Makefile*|Setup* ...')
        sys.exit(2)
    for arg in sys.argv[1:]:
        base = os.path.basename(arg)
        if base[:8] == 'Makefile':
            print('Make style parsing:', arg)
            v = getmakevars(arg)
            prdict(v)
        elif base[:5] == 'Setup':
            print('Setup style parsing:', arg)
            (m, v) = getsetupinfo(arg)
            prdict(m)
            prdict(v)
        else:
            print(arg, 'is neither a Makefile nor a Setup file')
            print('(name must begin with "Makefile" or "Setup")')

def prdict(d):
    if False:
        return 10
    keys = sorted(d.keys())
    for key in keys:
        value = d[key]
        print('%-15s' % key, str(value))
if __name__ == '__main__':
    test()