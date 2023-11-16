import os, sys, argparse, random
from shell import shell, ssh
description = 'Build release packages for pyqtgraph.'
epilog = '\nPackage build is done in several steps:\n\n    * Attempt to clone branch release-x.y.z from source-repo\n    * Merge release branch into master\n    * Write new version numbers into the source\n    * Roll over unreleased CHANGELOG entries\n    * Commit and tag new release\n    * Build HTML documentation\n    * Build source package\n    * Build deb packages (if running on Linux)\n    * Build Windows exe installers\n\nRelease packages may be published by using the --publish flag:\n\n    * Uploads release files to website\n    * Pushes tagged git commit to github\n    * Uploads source package to pypi\n\nBuilding source packages requires:\n\n    * \n    * \n    * python-sphinx\n\nBuilding deb packages requires several dependencies:\n\n    * build-essential\n    * python-all, python3-all\n    * python-stdeb, python3-stdeb\n    \nNote: building windows .exe files should be possible on any OS. However, \nDebian/Ubuntu systems do not include the necessary wininst*.exe files; these\nmust be manually copied from the Python source to the distutils/command \nsubmodule path (/usr/lib/pythonX.X/distutils/command). Additionally, it may be\nnecessary to rename (or copy / link) wininst-9.0-amd64.exe to \nwininst-6.0-amd64.exe.\n\n'
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
build_dir = os.path.join(path, 'release-build')
pkg_dir = os.path.join(path, 'release-packages')
ap = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('version', help='The x.y.z version to generate release packages for. There must be a corresponding pyqtgraph-x.y.z branch in the source repository.')
ap.add_argument('--publish', metavar='', help='Publish previously built package files (must be stored in pkg-dir/version) and tagged release commit (from build-dir).', action='store_const', const=True, default=False)
ap.add_argument('--source-repo', metavar='', help='Repository from which release and master branches will be cloned. Default is the repo containing this script.', default=path)
ap.add_argument('--build-dir', metavar='', help='Directory where packages will be staged and built. Default is source_root/release-build.', default=build_dir)
ap.add_argument('--pkg-dir', metavar='', help='Directory where packages will be stored. Default is source_root/release-packages.', default=pkg_dir)
ap.add_argument('--skip-pip-test', metavar='', help='Skip testing pip install.', action='store_const', const=True, default=False)
ap.add_argument('--no-deb', metavar='', help='Skip building Debian packages.', action='store_const', const=True, default=False)
ap.add_argument('--no-exe', metavar='', help='Skip building Windows exe installers.', action='store_const', const=True, default=False)

def build(args):
    if False:
        i = 10
        return i + 15
    if os.path.exists(args.build_dir):
        sys.stderr.write('Please remove the build directory %s before proceeding, or specify a different path with --build-dir.\n' % args.build_dir)
        sys.exit(-1)
    if os.path.exists(args.pkg_dir):
        sys.stderr.write('Please remove the package directory %s before proceeding, or specify a different path with --pkg-dir.\n' % args.pkg_dir)
        sys.exit(-1)
    shell('\n        # Clone and merge release branch into previous master\n        mkdir -p {build_dir}\n        cd {build_dir}\n        rm -rf pyqtgraph\n        git clone --depth 1 --branch master --single-branch {source_repo} pyqtgraph\n        cd pyqtgraph\n        git checkout -b release-{version}\n        git pull {source_repo} release-{version}\n        git checkout master\n        git merge --no-ff --no-commit release-{version}\n        \n        # Write new version number into the source\n        sed -i "s/__version__ = .*/__version__ = \'{version}\'/" pyqtgraph/__init__.py\n        sed -i "s/version = .*/version = \'{version}\'/" doc/source/conf.py\n        sed -i "s/release = .*/release = \'{version}\'/" doc/source/conf.py\n        \n        # make sure changelog mentions unreleased changes\n        grep "pyqtgraph-{version}.*unreleased.*" CHANGELOG    \n        sed -i "s/pyqtgraph-{version}.*unreleased.*/pyqtgraph-{version}/" CHANGELOG\n\n        # Commit and tag new release\n        git commit -a -m "PyQtGraph release {version}"\n        git tag pyqtgraph-{version}\n\n        # Build HTML documentation\n        cd doc\n            make clean\n            make html\n        cd ..\n        find ./ -name "*.pyc" -delete\n\n        # package source distribution\n        python setup.py sdist\n\n        mkdir -p {pkg_dir}\n        cp dist/*.tar.gz {pkg_dir}\n\n        # source package build complete.\n    '.format(**args.__dict__))
    if args.skip_pip_test:
        args.pip_test = 'skipped'
    else:
        shell('\n            # test pip install source distribution\n            rm -rf release-{version}-virtenv\n            virtualenv --system-site-packages release-{version}-virtenv\n            . release-{version}-virtenv/bin/activate\n            echo "PATH: $PATH"\n            echo "ENV: $VIRTUAL_ENV" \n            pip install --no-index --no-deps dist/pyqtgraph-{version}.tar.gz\n            deactivate\n            \n            # pip install test passed\n        '.format(**args.__dict__))
        args.pip_test = 'passed'
    if 'linux' in sys.platform and (not args.no_deb):
        shell('\n            # build deb packages\n            cd {build_dir}/pyqtgraph\n            python setup.py --command-packages=stdeb.command sdist_dsc\n            cd deb_dist/pyqtgraph-{version}\n            sed -i "s/^Depends:.*/Depends: python (>= 2.6), python-qt4 | python-pyside, python-numpy/" debian/control    \n            dpkg-buildpackage\n            cd ../../\n            mv deb_dist {pkg_dir}/pyqtgraph-{version}-deb\n            \n            # deb package build complete.\n        '.format(**args.__dict__))
        args.deb_status = 'built'
    else:
        args.deb_status = 'skipped'
    if not args.no_exe:
        shell('\n            # Build windows executables\n            cd {build_dir}/pyqtgraph\n            python setup.py build bdist_wininst --plat-name=win32\n            python setup.py build bdist_wininst --plat-name=win-amd64\n            cp dist/*.exe {pkg_dir}\n        '.format(**args.__dict__))
        args.exe_status = 'built'
    else:
        args.exe_status = 'skipped'
    print(unindent('\n\n    ======== Build complete. =========\n\n      * Source package:     built\n      * Pip install test:   {pip_test}\n      * Debian packages:    {deb_status}\n      * Windows installers: {exe_status}\n      * Package files in    {pkg_dir}\n\n    Next steps to publish:\n    \n      * Test all packages\n      * Run script again with --publish\n\n    ').format(**args.__dict__))

def publish(args):
    if False:
        print('Hello World!')
    if not os.path.isfile(os.path.expanduser('~/.pypirc')):
        print(unindent('\n            Missing ~/.pypirc file. Should look like:\n            -----------------------------------------\n\n                [distutils]\n                index-servers =\n                    pypi\n\n                [pypi]\n                username:your_username\n                password:your_password\n\n        '))
        sys.exit(-1)
    shell('\n        cd {build_dir}/pyqtgraph\n        \n        # Uploading documentation..  (disabled; now hosted by readthedocs.io)\n        #rsync -rv doc/build/* pyqtgraph.org:/www/code/pyqtgraph/pyqtgraph/documentation/build/\n\n        # Uploading release packages to website\n        rsync -v {pkg_dir} pyqtgraph.org:/www/code/pyqtgraph/downloads/\n\n        # Push master to github\n        git push https://github.com/pyqtgraph/pyqtgraph master:master\n        \n        # Push tag to github\n        git push https://github.com/pyqtgraph/pyqtgraph pyqtgraph-{version}\n\n        # Upload to pypi..\n        python setup.py sdist upload\n\n    '.format(**args.__dict__))
    print(unindent('\n\n    ======== Upload complete. =========\n\n    Next steps to publish:\n        - update website\n        - mailing list announcement\n        - new conda recipe (http://conda.pydata.org/docs/build.html)\n        - contact deb maintainer (gianfranco costamagna)\n        - other package maintainers?\n\n    ').format(**args.__dict__))

def unindent(msg):
    if False:
        while True:
            i = 10
    ind = 1000000.0
    lines = msg.split('\n')
    for line in lines:
        if len(line.strip()) == 0:
            continue
        ind = min(ind, len(line) - len(line.lstrip()))
    return '\n'.join([line[ind:] for line in lines])
if __name__ == '__main__':
    args = ap.parse_args()
    args.build_dir = os.path.abspath(args.build_dir)
    args.pkg_dir = os.path.join(os.path.abspath(args.pkg_dir), args.version)
    if args.publish:
        publish(args)
    else:
        build(args)