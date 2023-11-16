import fnmatch
import os
import sys
import verstamp
import win32api
numStamped = 0
g_patterns = ['*.dll', '*.pyd', '*.exe', '*.ocx']

def walk(arg, dirname, names):
    if False:
        i = 10
        return i + 15
    global numStamped
    (vars, debug, descriptions) = arg
    for name in names:
        for pat in g_patterns:
            if fnmatch.fnmatch(name, pat):
                pathname = os.path.join(dirname, name)
                (base, ext) = os.path.splitext(name)
                if base[-2:] == '_d':
                    name = base[:-2] + ext
                is_dll = ext.lower() != '.exe'
                if os.path.normcase(name) in descriptions:
                    desc = descriptions[os.path.normcase(name)]
                    try:
                        verstamp.stamp(vars, pathname, desc, is_dll=is_dll)
                        numStamped = numStamped + 1
                    except win32api.error as exc:
                        print('Could not stamp', pathname, 'Error', exc.winerror, '-', exc.strerror)
                else:
                    print('WARNING: description not provided for:', name)

def load_descriptions(fname, vars):
    if False:
        while True:
            i = 10
    retvars = {}
    descriptions = {}
    lines = open(fname, 'r').readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        if line != '' and line[0] != '#':
            idx1 = line.find(' ')
            idx2 = line.find('\t')
            if idx1 == -1 or idx2 < idx1:
                idx1 = idx2
            if idx1 == -1:
                print('ERROR: bad syntax in description file at line %d.' % (i + 1))
                sys.exit(1)
            key = line[:idx1]
            val = line[idx1:].strip()
            if key in vars:
                retvars[key] = val
            else:
                descriptions[key] = val
    if 'product' not in retvars:
        print('ERROR: description file is missing the product name.')
        sys.exit(1)
    if 'major' not in retvars:
        print('ERROR: description file is missing the major version number.')
        sys.exit(1)
    if 'minor' not in retvars:
        print('ERROR: description file is missing the minor version number.')
        sys.exit(1)
    return (retvars, descriptions)

def scan(build, root, desc, **custom_vars):
    if False:
        i = 10
        return i + 15
    global numStamped
    numStamped = 0
    try:
        build = int(build)
    except ValueError:
        print('ERROR: build number is not a number: %s' % build)
        sys.exit(1)
    debug = 0
    varList = ['major', 'minor', 'sub', 'company', 'copyright', 'trademarks', 'product']
    (vars, descriptions) = load_descriptions(desc, varList)
    vars['build'] = build
    vars.update(custom_vars)
    arg = (vars, debug, descriptions)
    os.path.walk(root, walk, arg)
    print('Stamped %d files.' % numStamped)
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("ERROR: incorrect invocation. See script's header comments.")
        sys.exit(1)
    scan(*tuple(sys.argv[1:]))