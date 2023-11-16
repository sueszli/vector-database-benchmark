"""
This paver file is intended to help with the release process as much as
possible. It relies on virtualenv to generate 'bootstrap' environments as
independent from the user system as possible (e.g. to make sure the sphinx doc
is built against the built numpy, not an installed one).

Building changelog + notes
==========================

Assumes you have git and the binaries/tarballs in installers/::

    paver write_release
    paver write_note

This automatically put the checksum into README.rst, and writes the Changelog.

TODO
====
    - the script is messy, lots of global variables
    - make it more easily customizable (through command line args)
    - missing targets: install & test, sdist test, debian packaging
    - fix bdist_mpkg: we build the same source twice -> how to make sure we use
      the same underlying python for egg install in venv and for bdist_mpkg
"""
import os
import sys
import shutil
import hashlib
import textwrap
import paver
from paver.easy import Bunch, options, task, sh
RELEASE_NOTES = 'doc/source/release/2.0.0-notes.rst'
options(installers=Bunch(releasedir='release', installersdir=os.path.join('release', 'installers')))
sys.path.insert(0, os.path.dirname(__file__))
try:
    from setup import FULLVERSION
finally:
    sys.path.pop(0)

def tarball_name(ftype='gztar'):
    if False:
        for i in range(10):
            print('nop')
    "Generate source distribution name\n\n    Parameters\n    ----------\n    ftype : {'zip', 'gztar'}\n        Type of archive, default is 'gztar'.\n\n    "
    root = f'numpy-{FULLVERSION}'
    if ftype == 'gztar':
        return root + '.tar.gz'
    elif ftype == 'zip':
        return root + '.zip'
    raise ValueError(f'Unknown type {type}')

@task
def sdist(options):
    if False:
        i = 10
        return i + 15
    'Make source distributions.\n\n    Parameters\n    ----------\n    options :\n        Set by ``task`` decorator.\n\n    '
    sh('git clean -xdf')
    sh('git submodule init')
    sh('git submodule update')
    sh('python3 setup.py sdist --formats=gztar,zip')
    idirs = options.installers.installersdir
    if not os.path.exists(idirs):
        os.makedirs(idirs)
    for ftype in ['gztar', 'zip']:
        source = os.path.join('dist', tarball_name(ftype))
        target = os.path.join(idirs, tarball_name(ftype))
        shutil.copy(source, target)

def _compute_hash(idirs, hashfunc):
    if False:
        return 10
    'Hash files using given hashfunc.\n\n    Parameters\n    ----------\n    idirs : directory path\n        Directory containing files to be hashed.\n    hashfunc : hash function\n        Function to be used to hash the files.\n\n    '
    released = paver.path.path(idirs).listdir()
    checksums = []
    for fpath in sorted(released):
        with open(fpath, 'rb') as fin:
            fhash = hashfunc(fin.read())
            checksums.append('%s  %s' % (fhash.hexdigest(), os.path.basename(fpath)))
    return checksums

def compute_md5(idirs):
    if False:
        for i in range(10):
            print('nop')
    'Compute md5 hash of files in idirs.\n\n    Parameters\n    ----------\n    idirs : directory path\n        Directory containing files to be hashed.\n\n    '
    return _compute_hash(idirs, hashlib.md5)

def compute_sha256(idirs):
    if False:
        return 10
    'Compute sha256 hash of files in idirs.\n\n    Parameters\n    ----------\n    idirs : directory path\n        Directory containing files to be hashed.\n\n    '
    return _compute_hash(idirs, hashlib.sha256)

def write_release_task(options, filename='README'):
    if False:
        return 10
    'Append hashes of release files to release notes.\n\n    This appends file hashes to the release notes and creates\n    four README files of the result in various formats:\n\n    - README.rst\n    - README.rst.gpg\n    - README.md\n    - README.md.gpg\n\n    The md file are created using `pandoc` so that the links are\n    properly updated. The gpg files are kept separate, so that\n    the unsigned files may be edited before signing if needed.\n\n    Parameters\n    ----------\n    options :\n        Set by ``task`` decorator.\n    filename : str\n        Filename of the modified notes. The file is written\n        in the release directory.\n\n    '
    idirs = options.installers.installersdir
    notes = paver.path.path(RELEASE_NOTES)
    rst_readme = paver.path.path(filename + '.rst')
    md_readme = paver.path.path(filename + '.md')
    with open(rst_readme, 'w') as freadme:
        with open(notes) as fnotes:
            freadme.write(fnotes.read())
        freadme.writelines(textwrap.dedent('\n            Checksums\n            =========\n\n            MD5\n            ---\n            ::\n\n            '))
        freadme.writelines([f'    {c}\n' for c in compute_md5(idirs)])
        freadme.writelines(textwrap.dedent('\n            SHA256\n            ------\n            ::\n\n            '))
        freadme.writelines([f'    {c}\n' for c in compute_sha256(idirs)])
    sh(f'pandoc -s -o {md_readme} {rst_readme}')
    if hasattr(options, 'gpg_key'):
        cmd = f'gpg --clearsign --armor --default_key {options.gpg_key}'
    else:
        cmd = 'gpg --clearsign --armor'
    sh(cmd + f' --output {rst_readme}.gpg {rst_readme}')
    sh(cmd + f' --output {md_readme}.gpg {md_readme}')

@task
def write_release(options):
    if False:
        while True:
            i = 10
    'Write the README files.\n\n    Two README files are generated from the release notes, one in ``rst``\n    markup for the general release, the other in ``md`` markup for the github\n    release notes.\n\n    Parameters\n    ----------\n    options :\n        Set by ``task`` decorator.\n\n    '
    rdir = options.installers.releasedir
    write_release_task(options, os.path.join(rdir, 'README'))