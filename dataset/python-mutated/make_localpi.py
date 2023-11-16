"""
Utility script to create a pypi-like directory structure (localpi)
from a number of Python packages in a directory of the local filesystem.

  DIRECTORY STRUCTURE (before):
      +-- downloads/
           +-- alice-1.0.zip
           +-- alice-1.0.tar.gz
           +-- bob-1.3.0.tar.gz
           +-- bob-1.4.2.tar.gz
           +-- charly-1.0.tar.bz2

  DIRECTORY STRUCTURE (afterwards):
      +-- downloads/
           +-- simple/
           |      +-- alice/index.html   --> ../../alice-*.*
           |      +-- bob/index.html     --> ../../bob-*.*
           |      +-- charly/index.html  --> ../../charly-*.*
           |      +-- index.html  --> alice/, bob/, ...
           +-- alice-1.0.zip
           +-- alice-1.0.tar.gz
           +-- bob-1.3.0.tar.gz
           +-- bob-1.4.2.tar.gz
           +-- charly-1.0.tar.bz2

USAGE EXAMPLE:

    mkdir -p /tmp/downloads
    pip install --download=/tmp/downloads argparse Jinja2
    make_localpi.py /tmp/downloads
    pip install --index-url=file:///tmp/downloads/simple argparse Jinja2

ALTERNATIVE:

    pip install --download=/tmp/downloads argparse Jinja2
    pip install --find-links=/tmp/downloads --no-index argparse Jinja2
"""
from __future__ import absolute_import, print_function, with_statement
from fnmatch import fnmatch
import os.path
import shutil
import sys
__author__ = 'Jens Engel'
__version__ = '0.2'
__license__ = 'BSD'
__copyright__ = '(c) 2013 by Jens Engel'

class Package(object):
    """
    Package entity that keeps track of:
      * one or more versions of this package
      * one or more archive types
    """
    PATTERNS = ['*.egg', '*.exe', '*.whl', '*.zip', '*.tar.gz', '*.tar.bz2', '*.7z']

    def __init__(self, filename, name=None):
        if False:
            for i in range(10):
                print('nop')
        if not name and filename:
            name = self.get_pkgname(filename)
        self.name = name
        self.files = []
        if filename:
            self.files.append(filename)

    @property
    def versions(self):
        if False:
            while True:
                i = 10
        versions_info = [self.get_pkgversion(p) for p in self.files]
        return versions_info

    @classmethod
    def split_pkgname_parts(cls, filename):
        if False:
            print('Hello World!')
        basename = cls.splitext(os.path.basename(filename))
        if basename.startswith('http') and '%2F' in basename:
            pos = basename.rfind('%2F')
            basename = basename[pos + 3:]
        version_part_index = 0
        parts = basename.split('-')
        for (index, part) in enumerate(parts):
            if index == 0:
                continue
            elif part and part[0].isdigit() and (len(part) >= 3):
                version_part_index = index
                break
        name = '-'.join(parts[:version_part_index])
        version = '0.0'
        remainder = None
        if version_part_index > 0:
            version = parts[version_part_index]
            if version_part_index + 1 < len(parts):
                remainder = '-'.join(parts[version_part_index + 1:])
        assert name, "OOPS: basename=%s, name='%s'" % (basename, name)
        return (name, version, remainder)

    @classmethod
    def get_pkgname(cls, filename):
        if False:
            while True:
                i = 10
        return cls.split_pkgname_parts(filename)[0]

    @classmethod
    def get_pkgversion(cls, filename):
        if False:
            while True:
                i = 10
        return cls.split_pkgname_parts(filename)[1]

    @classmethod
    def make_pkgname_with_version(cls, filename):
        if False:
            return 10
        pkg_name = cls.get_pkgname(filename)
        pkg_version = cls.get_pkgversion(filename)
        return '%s-%s' % (pkg_name, pkg_version)

    @staticmethod
    def splitext(filename):
        if False:
            i = 10
            return i + 15
        fname = os.path.splitext(filename)[0]
        if fname.endswith('.tar'):
            fname = os.path.splitext(fname)[0]
        return fname

    @classmethod
    def isa(cls, filename):
        if False:
            for i in range(10):
                print('nop')
        basename = os.path.basename(filename)
        if basename.startswith('.'):
            return False
        for pattern in cls.PATTERNS:
            if fnmatch(filename, pattern):
                return True
        return False

def collect_packages(package_dir, package_map=None):
    if False:
        i = 10
        return i + 15
    if package_map is None:
        package_map = {}
    packages = []
    for filename in sorted(os.listdir(package_dir)):
        if not Package.isa(filename):
            continue
        pkg_filepath = os.path.join(package_dir, filename)
        package_name = Package.get_pkgname(pkg_filepath)
        package = package_map.get(package_name, None)
        if not package:
            package = Package(pkg_filepath)
            package_map[package.name] = package
            packages.append(package)
        else:
            package.files.append(pkg_filepath)
    return packages

def make_index_for(package, index_dir, verbose=True):
    if False:
        i = 10
        return i + 15
    "\n    Create an 'index.html' for one package.\n\n    :param package:   Package object to use.\n    :param index_dir: Where 'index.html' should be created.\n    "
    index_template = '<html>\n<head><title>{title}</title></head>\n<body>\n<h1>{title}</h1>\n<ul>\n{packages}\n</ul>\n</body>\n</html>\n'
    item_template = '<li><a href="{1}">{0}</a></li>'
    index_filename = os.path.join(index_dir, 'index.html')
    if not os.path.isdir(index_dir):
        os.makedirs(index_dir)
    parts = []
    for pkg_filename in package.files:
        pkg_name = os.path.basename(pkg_filename)
        if pkg_name == 'index.html':
            pkg_name = os.path.basename(os.path.dirname(pkg_filename))
        else:
            pkg_name = package.make_pkgname_with_version(pkg_filename)
        pkg_relpath_to = os.path.relpath(pkg_filename, index_dir)
        parts.append(item_template.format(pkg_name, pkg_relpath_to))
    if not parts:
        print('OOPS: Package %s has no files' % package.name)
        return
    if verbose:
        root_index = not Package.isa(package.files[0])
        if root_index:
            info = 'with %d package(s)' % len(package.files)
        else:
            package_versions = sorted(set(package.versions))
            info = ', '.join(reversed(package_versions))
        message = '%-30s  %s' % (package.name, info)
        print(message)
    with open(index_filename, 'w') as f:
        packages = '\n'.join(parts)
        text = index_template.format(title=package.name, packages=packages)
        f.write(text.strip())
        f.close()

def make_package_index(download_dir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a pypi server like file structure below download directory.\n\n    :param download_dir:    Download directory with packages.\n\n    EXAMPLE BEFORE:\n      +-- downloads/\n           +-- wheelhouse/bob-1.4.2-*.whl\n           +-- alice-1.0.zip\n           +-- alice-1.0.tar.gz\n           +-- bob-1.3.0.tar.gz\n           +-- bob-1.4.2.tar.gz\n           +-- charly-1.0.tar.bz2\n\n    EXAMPLE AFTERWARDS:\n      +-- downloads/\n           +-- simple/\n           |      +-- alice/index.html   --> ../../alice-*.*\n           |      +-- bob/index.html     --> ../../bob-*.*\n           |      +-- charly/index.html  --> ../../charly-*.*\n           |      +-- index.html  --> alice/index.html, bob/index.html, ...\n           +-- wheelhouse/bob-1.4.2-*.whl\n           +-- alice-1.0.zip\n           +-- alice-1.0.tar.gz\n           +-- bob-1.3.0.tar.gz\n           +-- bob-1.4.2.tar.gz\n           +-- charly-1.0.tar.bz2\n    '
    if not os.path.isdir(download_dir):
        raise ValueError('No such directory: %r' % download_dir)
    pkg_rootdir = os.path.join(download_dir, 'simple')
    if os.path.isdir(pkg_rootdir):
        shutil.rmtree(pkg_rootdir, ignore_errors=True)
    os.mkdir(pkg_rootdir)
    package_dirs = [download_dir]
    wheelhouse_dir = os.path.join(download_dir, 'wheelhouse')
    if os.path.isdir(wheelhouse_dir):
        print('Using wheelhouse: %s' % wheelhouse_dir)
        package_dirs.append(wheelhouse_dir)
    package_map = {}
    packages = []
    for package_dir in package_dirs:
        new_packages = collect_packages(package_dir, package_map)
        packages.extend(new_packages)
    root_package = Package(None, 'Python Package Index')
    root_package.files = [os.path.join(pkg_rootdir, pkg.name, 'index.html') for pkg in packages]
    make_index_for(root_package, pkg_rootdir)
    for package in packages:
        index_dir = os.path.join(pkg_rootdir, package.name)
        make_index_for(package, index_dir)
if __name__ == '__main__':
    if len(sys.argv) != 2 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        print('USAGE: %s DOWNLOAD_DIR' % os.path.basename(sys.argv[0]))
        print(__doc__)
        sys.exit(1)
    make_package_index(sys.argv[1])