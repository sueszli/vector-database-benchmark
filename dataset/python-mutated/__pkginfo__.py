"""uncompyle6 packaging information"""
copyright = '\nCopyright (C) 2015-2021 Rocky Bernstein <rb@dustyfeet.com>.\n'
classifiers = ['Development Status :: 5 - Production/Stable', 'Intended Audience :: Developers', 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', 'Operating System :: OS Independent', 'Programming Language :: Python', 'Programming Language :: Python :: 2', 'Programming Language :: Python :: 2.4', 'Programming Language :: Python :: 2.5', 'Programming Language :: Python :: 2.6', 'Programming Language :: Python :: 2.7', 'Programming Language :: Python :: 3', 'Programming Language :: Python :: 3.0', 'Programming Language :: Python :: 3.1', 'Programming Language :: Python :: 3.2', 'Programming Language :: Python :: 3.3', 'Programming Language :: Python :: 3.4', 'Programming Language :: Python :: 3.5', 'Programming Language :: Python :: 3.6', 'Programming Language :: Python :: 3.7', 'Programming Language :: Python :: 3.8', 'Programming Language :: Python :: 3.9', 'Programming Language :: Python :: 3.10', 'Programming Language :: Python :: Implementation :: PyPy', 'Topic :: Software Development :: Debuggers', 'Topic :: Software Development :: Libraries :: Python Modules']
author = 'Rocky Bernstein, Hartmut Goebel, John Aycock, and others'
author_email = 'rb@dustyfeet.com'
entry_points = {'console_scripts': ['uncompyle6=uncompyle6.bin.uncompile:main_bin', 'pydisassemble=uncompyle6.bin.pydisassemble:main']}
ftp_url = None
install_requires = ['spark-parser >= 1.8.9, < 1.9.0', 'xdis >= 6.0.8, < 6.2.0']
license = 'GPL3'
mailing_list = 'python-debugger@googlegroups.com'
modname = 'uncompyle6'
py_modules = None
short_desc = 'Python cross-version byte-code decompiler'
web = 'https://github.com/rocky/python-uncompyle6/'
zip_safe = True
import os.path

def get_srcdir():
    if False:
        for i in range(10):
            print('nop')
    filename = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    return os.path.realpath(filename)
srcdir = get_srcdir()

def read(*rnames):
    if False:
        for i in range(10):
            print('nop')
    return open(os.path.join(srcdir, *rnames)).read()
long_description = read('README.rst') + '\n'
exec(read('uncompyle6/version.py'))