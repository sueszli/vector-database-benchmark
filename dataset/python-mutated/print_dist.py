"""List and pretty print tarball & wheel files in the dist/ directory."""
import argparse
import collections
import os
from psutil._common import bytes2human
from psutil._common import print_color

class Wheel:

    def __init__(self, path):
        if False:
            print('Hello World!')
        self._path = path
        self._name = os.path.basename(path)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s(name=%s, plat=%s, arch=%s, pyver=%s)>' % (self.__class__.__name__, self.name, self.platform(), self.arch(), self.pyver())
    __str__ = __repr__

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self._name

    def platform(self):
        if False:
            i = 10
            return i + 15
        plat = self.name.split('-')[-1]
        pyimpl = self.name.split('-')[3]
        ispypy = 'pypy' in pyimpl
        if 'linux' in plat:
            if ispypy:
                return 'pypy_on_linux'
            else:
                return 'linux'
        elif 'win' in plat:
            if ispypy:
                return 'pypy_on_windows'
            else:
                return 'windows'
        elif 'macosx' in plat:
            if ispypy:
                return 'pypy_on_macos'
            else:
                return 'macos'
        else:
            raise ValueError('unknown platform %r' % self.name)

    def arch(self):
        if False:
            for i in range(10):
                print('nop')
        if self.name.endswith(('x86_64.whl', 'amd64.whl')):
            return '64'
        if self.name.endswith(('i686.whl', 'win32.whl')):
            return '32'
        if self.name.endswith('arm64.whl'):
            return 'arm64'
        return '?'

    def pyver(self):
        if False:
            print('Hello World!')
        pyver = 'pypy' if self.name.split('-')[3].startswith('pypy') else 'py'
        pyver += self.name.split('-')[2][2:]
        return pyver

    def size(self):
        if False:
            return 10
        return os.path.getsize(self._path)

class Tarball(Wheel):

    def platform(self):
        if False:
            for i in range(10):
                print('nop')
        return 'source'

    def arch(self):
        if False:
            print('Hello World!')
        return '-'

    def pyver(self):
        if False:
            for i in range(10):
                print('nop')
        return '-'

def main():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', nargs='?', default='dist', help='directory containing tar.gz or wheel files')
    args = parser.parse_args()
    groups = collections.defaultdict(list)
    ls = sorted(os.listdir(args.dir), key=lambda x: x.endswith('tar.gz'))
    for name in ls:
        path = os.path.join(args.dir, name)
        if path.endswith('.whl'):
            pkg = Wheel(path)
        elif path.endswith('.tar.gz'):
            pkg = Tarball(path)
        else:
            raise ValueError('invalid package %r' % path)
        groups[pkg.platform()].append(pkg)
    tot_files = 0
    tot_size = 0
    templ = '%-100s %7s %7s %7s'
    for (platf, pkgs) in groups.items():
        ppn = '%s (%s)' % (platf, len(pkgs))
        s = templ % (ppn, 'size', 'arch', 'pyver')
        print_color('\n' + s, color=None, bold=True)
        for pkg in sorted(pkgs, key=lambda x: x.name):
            tot_files += 1
            tot_size += pkg.size()
            s = templ % ('  ' + pkg.name, bytes2human(pkg.size()), pkg.arch(), pkg.pyver())
            if 'pypy' in pkg.pyver():
                print_color(s, color='violet')
            else:
                print_color(s, color='brown')
    print_color('\n\ntotals: files=%s, size=%s' % (tot_files, bytes2human(tot_size)), bold=True)
if __name__ == '__main__':
    main()