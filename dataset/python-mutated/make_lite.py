"""
Usage: make_lite.py <wrapped_routines_file> <lapack_dir>

Typical invocation:

    make_lite.py wrapped_routines /tmp/lapack-3.x.x

Requires the following to be on the path:
 * f2c
 * patch

"""
import sys
import os
import re
import subprocess
import shutil
import fortran
import clapack_scrub
try:
    from distutils.spawn import find_executable as which
except ImportError:
    from shutil import which
F2C_ARGS = ['-A', '-Nx800']
HEADER_BLURB = '/*\n * NOTE: This is generated code. Look in numpy/linalg/lapack_lite for\n *       information on remaking this file.\n */\n'
HEADER = HEADER_BLURB + '#include "f2c.h"\n\n#ifdef HAVE_CONFIG\n#include "config.h"\n#else\nextern doublereal dlamch_(char *);\n#define EPSILON dlamch_("Epsilon")\n#define SAFEMINIMUM dlamch_("Safe minimum")\n#define PRECISION dlamch_("Precision")\n#define BASE dlamch_("Base")\n#endif\n\nextern doublereal dlapy2_(doublereal *x, doublereal *y);\n\n/*\nf2c knows the exact rules for precedence, and so omits parentheses where not\nstrictly necessary. Since this is generated code, we don\'t really care if\nit\'s readable, and we know what is written is correct. So don\'t warn about\nthem.\n*/\n#if defined(__GNUC__)\n#pragma GCC diagnostic ignored "-Wparentheses"\n#endif\n'

class FortranRoutine:
    """Wrapper for a Fortran routine in a file.
    """
    type = 'generic'

    def __init__(self, name=None, filename=None):
        if False:
            print('Hello World!')
        self.filename = filename
        if name is None:
            (root, ext) = os.path.splitext(filename)
            name = root
        self.name = name
        self._dependencies = None

    def dependencies(self):
        if False:
            i = 10
            return i + 15
        if self._dependencies is None:
            deps = fortran.getDependencies(self.filename)
            self._dependencies = [d.lower() for d in deps]
        return self._dependencies

    def __repr__(self):
        if False:
            return 10
        return 'FortranRoutine({!r}, filename={!r})'.format(self.name, self.filename)

class UnknownFortranRoutine(FortranRoutine):
    """Wrapper for a Fortran routine for which the corresponding file
    is not known.
    """
    type = 'unknown'

    def __init__(self, name):
        if False:
            return 10
        FortranRoutine.__init__(self, name=name, filename='<unknown>')

    def dependencies(self):
        if False:
            for i in range(10):
                print('nop')
        return []

class FortranLibrary:
    """Container for a bunch of Fortran routines.
    """

    def __init__(self, src_dirs):
        if False:
            for i in range(10):
                print('nop')
        self._src_dirs = src_dirs
        self.names_to_routines = {}

    def _findRoutine(self, rname):
        if False:
            return 10
        rname = rname.lower()
        for s in self._src_dirs:
            ffilename = os.path.join(s, rname + '.f')
            if os.path.exists(ffilename):
                return self._newFortranRoutine(rname, ffilename)
        return UnknownFortranRoutine(rname)

    def _newFortranRoutine(self, rname, filename):
        if False:
            i = 10
            return i + 15
        return FortranRoutine(rname, filename)

    def addIgnorableRoutine(self, rname):
        if False:
            for i in range(10):
                print('nop')
        "Add a routine that we don't want to consider when looking at\n        dependencies.\n        "
        rname = rname.lower()
        routine = UnknownFortranRoutine(rname)
        self.names_to_routines[rname] = routine

    def addRoutine(self, rname):
        if False:
            for i in range(10):
                print('nop')
        'Add a routine to the library.\n        '
        self.getRoutine(rname)

    def getRoutine(self, rname):
        if False:
            return 10
        "Get a routine from the library. Will add if it's not found.\n        "
        unique = []
        rname = rname.lower()
        routine = self.names_to_routines.get(rname, unique)
        if routine is unique:
            routine = self._findRoutine(rname)
            self.names_to_routines[rname] = routine
        return routine

    def allRoutineNames(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the names of all the routines.\n        '
        return list(self.names_to_routines.keys())

    def allRoutines(self):
        if False:
            for i in range(10):
                print('nop')
        'Return all the routines.\n        '
        return list(self.names_to_routines.values())

    def resolveAllDependencies(self):
        if False:
            print('Hello World!')
        'Try to add routines to the library to satisfy all the dependencies\n        for each routine in the library.\n\n        Returns a set of routine names that have the dependencies unresolved.\n        '
        done_this = set()
        last_todo = set()
        while True:
            todo = set(self.allRoutineNames()) - done_this
            if todo == last_todo:
                break
            for rn in todo:
                r = self.getRoutine(rn)
                deps = r.dependencies()
                for d in deps:
                    self.addRoutine(d)
                done_this.add(rn)
            last_todo = todo
        return todo

class LapackLibrary(FortranLibrary):

    def _newFortranRoutine(self, rname, filename):
        if False:
            print('Hello World!')
        routine = FortranLibrary._newFortranRoutine(self, rname, filename)
        if 'blas' in filename.lower():
            routine.type = 'blas'
        elif 'install' in filename.lower():
            routine.type = 'config'
        elif rname.startswith('z'):
            routine.type = 'z_lapack'
        elif rname.startswith('c'):
            routine.type = 'c_lapack'
        elif rname.startswith('s'):
            routine.type = 's_lapack'
        elif rname.startswith('d'):
            routine.type = 'd_lapack'
        else:
            routine.type = 'lapack'
        return routine

    def allRoutinesByType(self, typename):
        if False:
            for i in range(10):
                print('nop')
        routines = sorted(((r.name, r) for r in self.allRoutines() if r.type == typename))
        return [a[1] for a in routines]

def printRoutineNames(desc, routines):
    if False:
        while True:
            i = 10
    print(desc)
    for r in routines:
        print('\t%s' % r.name)

def getLapackRoutines(wrapped_routines, ignores, lapack_dir):
    if False:
        for i in range(10):
            print('nop')
    blas_src_dir = os.path.join(lapack_dir, 'BLAS', 'SRC')
    if not os.path.exists(blas_src_dir):
        blas_src_dir = os.path.join(lapack_dir, 'blas', 'src')
    lapack_src_dir = os.path.join(lapack_dir, 'SRC')
    if not os.path.exists(lapack_src_dir):
        lapack_src_dir = os.path.join(lapack_dir, 'src')
    install_src_dir = os.path.join(lapack_dir, 'INSTALL')
    if not os.path.exists(install_src_dir):
        install_src_dir = os.path.join(lapack_dir, 'install')
    library = LapackLibrary([install_src_dir, blas_src_dir, lapack_src_dir])
    for r in ignores:
        library.addIgnorableRoutine(r)
    for w in wrapped_routines:
        library.addRoutine(w)
    library.resolveAllDependencies()
    return library

def getWrappedRoutineNames(wrapped_routines_file):
    if False:
        i = 10
        return i + 15
    routines = []
    ignores = []
    with open(wrapped_routines_file) as fo:
        for line in fo:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('IGNORE:'):
                line = line[7:].strip()
                ig = line.split()
                ignores.extend(ig)
            else:
                routines.append(line)
    return (routines, ignores)
types = {'blas', 'lapack', 'd_lapack', 's_lapack', 'z_lapack', 'c_lapack', 'config'}

def dumpRoutineNames(library, output_dir):
    if False:
        for i in range(10):
            print('nop')
    for typename in {'unknown'} | types:
        routines = library.allRoutinesByType(typename)
        filename = os.path.join(output_dir, typename + '_routines.lst')
        with open(filename, 'w') as fo:
            for r in routines:
                deps = r.dependencies()
                fo.write('%s: %s\n' % (r.name, ' '.join(deps)))

def concatenateRoutines(routines, output_file):
    if False:
        return 10
    with open(output_file, 'w') as output_fo:
        for r in routines:
            with open(r.filename) as fo:
                source = fo.read()
            output_fo.write(source)

class F2CError(Exception):
    pass

def runF2C(fortran_filename, output_dir):
    if False:
        while True:
            i = 10
    fortran_filename = fortran_filename.replace('\\', '/')
    try:
        subprocess.check_call(['f2c'] + F2C_ARGS + ['-d', output_dir, fortran_filename])
    except subprocess.CalledProcessError:
        raise F2CError

def scrubF2CSource(c_file):
    if False:
        i = 10
        return i + 15
    with open(c_file) as fo:
        source = fo.read()
    source = clapack_scrub.scrubSource(source, verbose=True)
    with open(c_file, 'w') as fo:
        fo.write(HEADER)
        fo.write(source)

def ensure_executable(name):
    if False:
        return 10
    try:
        which(name)
    except Exception:
        raise SystemExit(name + ' not found')

def create_name_header(output_dir):
    if False:
        for i in range(10):
            print('nop')
    routine_re = re.compile('^      (subroutine|.* function)\\s+(\\w+)\\(.*$', re.I)
    extern_re = re.compile('^extern [a-z]+ ([a-z0-9_]+)\\(.*$')
    symbols = set(['xerbla'])
    for fn in os.listdir(output_dir):
        fn = os.path.join(output_dir, fn)
        if not fn.endswith('.f'):
            continue
        with open(fn) as f:
            for line in f:
                m = routine_re.match(line)
                if m:
                    symbols.add(m.group(2).lower())
    f2c_symbols = set()
    with open('f2c.h') as f:
        for line in f:
            m = extern_re.match(line)
            if m:
                f2c_symbols.add(m.group(1))
    with open(os.path.join(output_dir, 'lapack_lite_names.h'), 'w') as f:
        f.write(HEADER_BLURB)
        f.write("/*\n * This file renames all BLAS/LAPACK and f2c symbols to avoid\n * dynamic symbol name conflicts, in cases where e.g.\n * integer sizes do not match with 'standard' ABI.\n */\n")
        for name in sorted(symbols):
            f.write('#define %s_ BLAS_FUNC(%s)\n' % (name, name))
        f.write('\n/* Symbols exported by f2c.c */\n')
        for name in sorted(f2c_symbols):
            f.write('#define %s numpy_lapack_lite_%s\n' % (name, name))

def main():
    if False:
        for i in range(10):
            print('nop')
    if len(sys.argv) != 3:
        print(__doc__)
        return
    ensure_executable('f2c')
    ensure_executable('patch')
    wrapped_routines_file = sys.argv[1]
    lapack_src_dir = sys.argv[2]
    output_dir = os.path.join(os.path.dirname(__file__), 'build')
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
    (wrapped_routines, ignores) = getWrappedRoutineNames(wrapped_routines_file)
    library = getLapackRoutines(wrapped_routines, ignores, lapack_src_dir)
    dumpRoutineNames(library, output_dir)
    for typename in types:
        fortran_file = os.path.join(output_dir, 'f2c_%s.f' % typename)
        c_file = fortran_file[:-2] + '.c'
        print('creating %s ...' % c_file)
        routines = library.allRoutinesByType(typename)
        concatenateRoutines(routines, fortran_file)
        patch_file = os.path.basename(fortran_file) + '.patch'
        if os.path.exists(patch_file):
            subprocess.check_call(['patch', '-u', fortran_file, patch_file])
            print('Patched {}'.format(fortran_file))
        try:
            runF2C(fortran_file, output_dir)
        except F2CError:
            print('f2c failed on %s' % fortran_file)
            break
        scrubF2CSource(c_file)
        c_patch_file = os.path.basename(c_file) + '.patch'
        if os.path.exists(c_patch_file):
            subprocess.check_call(['patch', '-u', c_file, c_patch_file])
        print()
    create_name_header(output_dir)
    for fname in os.listdir(output_dir):
        if fname.endswith('.c') or fname == 'lapack_lite_names.h':
            print('Copying ' + fname)
            shutil.copy(os.path.join(output_dir, fname), os.path.abspath(os.path.dirname(__file__)))
if __name__ == '__main__':
    main()