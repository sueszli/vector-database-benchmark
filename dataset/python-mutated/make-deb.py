"""
Create a Debian Package.

Required dependencies:
  sudo apt-get install python-support
  sudo apt-get install python-pip
  sudo pip install stdeb==0.6.0
"""
import subprocess
import os, sys
import platform
import argparse
import re
import shutil
BITS = platform.architecture()[0]
assert BITS == '32bit' or BITS == '64bit'
ARCHITECTURE = 'i386' if BITS == '32bit' else 'amd64'
if BITS == '32bit':
    LINUX_BITS = 'linux32'
else:
    LINUX_BITS = 'linux64'
PACKAGE_NAME = 'cefpython3'
PYTHON_NAME = 'python2.7'
HOMEPAGE = 'https://code.google.com/p/cefpython/'
MAINTAINER = 'Czarek Tomczak <czarek.tomczak@gmail.com>'
DESCRIPTION_EXTENDED = ' CEF Python 3 is a python library for embedding the Chromium\n browser. It uses the Chromium Embedded Framework (CEF) internally.\n Examples of embedding are available for many GUI toolkits,\n including wxPython, PyGTK, PyQt, PySide, Kivy and PyWin32.\n'
COPYRIGHT = 'Format: http://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\nName: cefpython3\nMaintainer: %s\nSource: %s\n\nCopyright: 2012-2014 The CEF Python authors\nLicense: BSD 3-Clause\n\nFiles: *\nCopyright: 2012-2014 The CEF Python authors\nLicense: BSD 3-Clause\n' % (MAINTAINER, HOMEPAGE)
PYTHON_VERSION_WITH_DOT = str(sys.version_info.major) + '.' + str(sys.version_info.minor)
VERSION = None
INSTALLER = os.path.dirname(os.path.abspath(__file__))
DISTUTILS_SETUP = None
DEB_DIST = None
DEB_DIST_PACKAGE = None
DEBIAN = None
DEBIAN_PACKAGE = None

def log(msg):
    if False:
        while True:
            i = 10
    print('[make-deb.py] %s' % msg)

def replace_in_file(f_path, s_what, s_with):
    if False:
        while True:
            i = 10
    contents = ''
    with open(f_path, 'r') as f:
        contents = f.read()
    assert contents, 'Failed reading file: %s' % f_path
    contents = contents.replace(s_what, s_with)
    with open(f_path, 'w') as f:
        f.write(contents)

def remove_directories_from_previous_run():
    if False:
        return 10
    if os.path.exists(DISTUTILS_SETUP):
        log('The Distutils setup directory already exists, removing..')
        shutil.rmtree(DISTUTILS_SETUP)
    if os.path.exists(INSTALLER + '/deb_dist/'):
        log('The deb_dist/ directory already exists, removing..')
        shutil.rmtree(INSTALLER + '/deb_dist/')
    if os.path.exists(INSTALLER + '/deb_archive/'):
        log('The deb_archive/ directory already exists, removing..')
        shutil.rmtree(INSTALLER + '/deb_archive/')

def create_distutils_setup_package():
    if False:
        print('Hello World!')
    log('Creating Distutils setup package')
    subprocess.call('%s %s/make-setup.py -v %s' % (sys.executable, INSTALLER, VERSION), shell=True)
    assert os.path.exists(DISTUTILS_SETUP), 'Distutils Setup directory not found'

def modify_control_file():
    if False:
        print('Hello World!')
    log('Modyfing debian control file')
    control_file = DEBIAN + '/control'
    with open(control_file, 'r') as f:
        contents = f.read()
    contents = contents.replace('Architecture: all', 'Architecture: %s' % ARCHITECTURE)
    contents = re.sub('[\r\n]+$', '', contents)
    contents += '\n'
    description = DESCRIPTION_EXTENDED
    description = re.sub('[\r\n]+', '\n', description)
    description = re.sub('\n$', '', description)
    contents += '%s\n' % description
    contents += 'Version: %s-1\n' % VERSION
    contents += 'Maintainer: %s\n' % MAINTAINER
    contents += 'Homepage: %s\n' % HOMEPAGE
    contents += '\n'
    with open(control_file, 'w') as f:
        f.write(contents)

def create_copyright_file():
    if False:
        print('Hello World!')
    log('Creating debian copyright file')
    copyright = COPYRIGHT
    copyright = re.sub('[\r\n]', '\n', copyright)
    copyright += '\n'
    copyright += 'License: BSD 3-clause\n'
    with open(INSTALLER + '/../../../License', 'r') as f:
        license = f.readlines()
    for line in license:
        if not len(re.sub('\\s+', '', line)):
            copyright += ' .\n'
        else:
            copyright += ' ' + line.rstrip() + '\n'
    copyright += '\n'
    with open(DEBIAN + '/copyright', 'w') as f:
        f.write(copyright)

def copy_postinst_script():
    if False:
        for i in range(10):
            print('nop')
    log('Copying .postinst script')
    shutil.copy(INSTALLER + '/debian.postinst', DEBIAN + '/python-%s.postinst' % PACKAGE_NAME)

def create_debian_source_package():
    if False:
        for i in range(10):
            print('nop')
    log('Creating Debian source package using stdeb')
    os.chdir(DISTUTILS_SETUP)
    shutil.copy('../stdeb.cfg.template', 'stdeb.cfg')
    stdeb_cfg_add_deps('stdeb.cfg')
    subprocess.call('%s setup.py --command-packages=stdeb.command sdist_dsc' % (sys.executable,), shell=True)

def stdeb_cfg_add_deps(stdeb_cfg):
    if False:
        print('Hello World!')
    log('Adding deps to stdeb.cfg')
    with open(INSTALLER + '/deps.txt', 'r') as f:
        deps = f.read()
    deps = deps.strip()
    deps = deps.splitlines()
    for (i, dep) in enumerate(deps):
        deps[i] = dep.strip()
    deps = ', '.join(deps)
    with open(stdeb_cfg, 'a') as f:
        f.write('\nDepends: %s' % deps)

def deb_dist_cleanup():
    if False:
        while True:
            i = 10
    log('Preparing the deb_dist directory')
    os.system('mv %s %s' % (DISTUTILS_SETUP + '/deb_dist', INSTALLER + '/deb_dist'))
    os.system('rm -rf %s' % DISTUTILS_SETUP)
    global DEB_DIST, DEB_DIST_PACKAGE, DEBIAN, DEBIAN_PACKAGE
    DEB_DIST = INSTALLER + '/deb_dist'
    DEB_DIST_PACKAGE = DEB_DIST + '/' + PACKAGE_NAME + '-' + VERSION
    DEBIAN = DEB_DIST_PACKAGE + '/debian'
    DEBIAN_PACKAGE = DEBIAN + '/python-' + PACKAGE_NAME
    os.chdir(DEB_DIST)
    os.system('rm *.gz')
    os.system('rm *.dsc')

def create_debian_binary_package():
    if False:
        i = 10
        return i + 15
    os.chdir(DEB_DIST_PACKAGE)
    subprocess.call('dpkg-buildpackage -rfakeroot -uc -us', shell=True)

def modify_deb_archive():
    if False:
        while True:
            i = 10
    log('Modifying the deb archive')
    deb_archive_name = 'python-%s_%s-1_%s.deb' % (PACKAGE_NAME, VERSION, ARCHITECTURE)
    deb_archive_dir = INSTALLER + '/deb_archive'
    log('Moving the deb archive')
    os.system('mkdir %s' % deb_archive_dir)
    os.system('mv %s %s' % (DEB_DIST + '/' + deb_archive_name, deb_archive_dir + '/' + deb_archive_name))
    os.system('rm -rf %s' % DEB_DIST)
    log('Extracting the deb archive')
    os.chdir(deb_archive_dir)
    os.system('dpkg-deb -x %s .' % deb_archive_name)
    os.system('dpkg-deb -e %s' % deb_archive_name)
    os.system('rm %s' % deb_archive_name)
    log('Moving the .so libraries')
    lib_pyshared = './usr/lib/pyshared/%s/%s' % (PYTHON_NAME, PACKAGE_NAME)
    share_pyshared = './usr/share/pyshared/%s' % PACKAGE_NAME
    os.system('mv %s/*.so %s/' % (lib_pyshared, share_pyshared))
    os.system('rm -rf ./usr/lib/')
    log('Modifying paths in the text files')
    old_path = 'usr/lib/pyshared/%s/%s/' % (PYTHON_NAME, PACKAGE_NAME)
    new_path = 'usr/share/pyshared/%s/' % PACKAGE_NAME
    md5sums_file = './DEBIAN/md5sums'
    cefpython3_public_file = './usr/share/python-support/python-%s.public' % PACKAGE_NAME
    old_md5sum = subprocess.check_output('md5sum %s | cut -c 1-32' % cefpython3_public_file, shell=True).strip()
    replace_in_file(md5sums_file, old_path, new_path)
    replace_in_file(cefpython3_public_file, old_path, new_path)
    new_md5sum = subprocess.check_output('md5sum %s | cut -c 1-32' % cefpython3_public_file, shell=True).strip()
    replace_in_file(md5sums_file, old_md5sum, new_md5sum)
    log('Creating deb archive from the modified files')
    os.system('fakeroot dpkg-deb -b . ./%s' % deb_archive_name)

def main():
    if False:
        return 10
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')
    parser.add_argument('-v', '--version', help='cefpython version', required=True)
    args = parser.parse_args()
    assert re.search('^\\d+\\.\\d+$', args.version), 'Invalid version string'
    global VERSION
    VERSION = args.version
    global DISTUTILS_SETUP
    DISTUTILS_SETUP = INSTALLER + '/' + PACKAGE_NAME + '-' + args.version + '-' + LINUX_BITS + '-setup'
    remove_directories_from_previous_run()
    create_distutils_setup_package()
    create_debian_source_package()
    deb_dist_cleanup()
    modify_control_file()
    create_copyright_file()
    copy_postinst_script()
    create_debian_binary_package()
    modify_deb_archive()
    log('DONE')
if __name__ == '__main__':
    main()