import os.path
import subprocess
import sys
import sysconfig
ALLOWED_PREFIXES = ('Py', '_Py')
if sys.platform == 'darwin':
    ALLOWED_PREFIXES += ('__Py',)
IGNORED_EXTENSION = '_ctypes_test'
IGNORED_SYMBOLS = {'_init', '_fini'}

def is_local_symbol_type(symtype):
    if False:
        print('Hello World!')
    if symtype.islower() and symtype not in 'uvw':
        return True
    if symtype in 'bBdD':
        return True
    return False

def get_exported_symbols(library, dynamic=False):
    if False:
        while True:
            i = 10
    print(f'Check that {library} only exports symbols starting with Py or _Py')
    args = ['nm', '--no-sort']
    if dynamic:
        args.append('--dynamic')
    args.append(library)
    print('+ %s' % ' '.join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, universal_newlines=True)
    if proc.returncode:
        sys.stdout.write(proc.stdout)
        sys.exit(proc.returncode)
    stdout = proc.stdout.rstrip()
    if not stdout:
        raise Exception('command output is empty')
    return stdout

def get_smelly_symbols(stdout):
    if False:
        for i in range(10):
            print('nop')
    smelly_symbols = []
    python_symbols = []
    local_symbols = []
    for line in stdout.splitlines():
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        symtype = parts[1].strip()
        symbol = parts[-1]
        result = '%s (type: %s)' % (symbol, symtype)
        if symbol.startswith(ALLOWED_PREFIXES):
            python_symbols.append(result)
            continue
        if is_local_symbol_type(symtype):
            local_symbols.append(result)
        elif symbol in IGNORED_SYMBOLS:
            local_symbols.append(result)
        else:
            smelly_symbols.append(result)
    if local_symbols:
        print(f'Ignore {len(local_symbols)} local symbols')
    return (smelly_symbols, python_symbols)

def check_library(library, dynamic=False):
    if False:
        for i in range(10):
            print('nop')
    nm_output = get_exported_symbols(library, dynamic)
    (smelly_symbols, python_symbols) = get_smelly_symbols(nm_output)
    if not smelly_symbols:
        print(f'OK: no smelly symbol found ({len(python_symbols)} Python symbols)')
        return 0
    print()
    smelly_symbols.sort()
    for symbol in smelly_symbols:
        print('Smelly symbol: %s' % symbol)
    print()
    print('ERROR: Found %s smelly symbols!' % len(smelly_symbols))
    return len(smelly_symbols)

def check_extensions():
    if False:
        while True:
            i = 10
    print(__file__)
    config_dir = os.path.dirname(sysconfig.get_config_h_filename())
    filename = os.path.join(config_dir, 'pybuilddir.txt')
    try:
        with open(filename, encoding='utf-8') as fp:
            pybuilddir = fp.readline()
    except FileNotFoundError:
        print(f'Cannot check extensions because {filename} does not exist')
        return True
    print(f'Check extension modules from {pybuilddir} directory')
    builddir = os.path.join(config_dir, pybuilddir)
    nsymbol = 0
    for name in os.listdir(builddir):
        if not name.endswith('.so'):
            continue
        if IGNORED_EXTENSION in name:
            print()
            print(f'Ignore extension: {name}')
            continue
        print()
        filename = os.path.join(builddir, name)
        nsymbol += check_library(filename, dynamic=True)
    return nsymbol

def main():
    if False:
        for i in range(10):
            print('nop')
    nsymbol = 0
    LIBRARY = sysconfig.get_config_var('LIBRARY')
    if not LIBRARY:
        raise Exception('failed to get LIBRARY variable from sysconfig')
    if os.path.exists(LIBRARY):
        nsymbol += check_library(LIBRARY)
    LDLIBRARY = sysconfig.get_config_var('LDLIBRARY')
    if not LDLIBRARY:
        raise Exception('failed to get LDLIBRARY variable from sysconfig')
    if LDLIBRARY != LIBRARY:
        print()
        nsymbol += check_library(LDLIBRARY, dynamic=True)
    nsymbol += check_extensions()
    if nsymbol:
        print()
        print(f'ERROR: Found {nsymbol} smelly symbols in total!')
        sys.exit(1)
    print()
    print(f"OK: all exported symbols of all libraries are prefixed with {' or '.join(map(repr, ALLOWED_PREFIXES))}")
if __name__ == '__main__':
    main()