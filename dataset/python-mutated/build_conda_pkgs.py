import os
import re
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from configparser import ConfigParser
from datetime import timedelta
from logging import Formatter, StreamHandler, getLogger
from pathlib import Path
from subprocess import check_call
from textwrap import dedent
from time import time
from git import Repo, rmtree
from ruamel.yaml import YAML
from setuptools_scm import get_version
fmt = Formatter('%(asctime)s [%(levelname)s] [%(name)s] -> %(message)s')
h = StreamHandler()
h.setFormatter(fmt)
logger = getLogger('BuildCondaPkgs')
logger.addHandler(h)
logger.setLevel('INFO')
HERE = Path(__file__).parent
BUILD = HERE / 'build'
RESOURCES = HERE / 'resources'
EXTDEPS = HERE.parent / 'external-deps'
SPECS = BUILD / 'specs.yaml'
REQUIREMENTS = HERE.parent / 'requirements'
REQ_MAIN = REQUIREMENTS / 'main.yml'
REQ_WINDOWS = REQUIREMENTS / 'windows.yml'
REQ_MAC = REQUIREMENTS / 'macos.yml'
REQ_LINUX = REQUIREMENTS / 'linux.yml'
BUILD.mkdir(exist_ok=True)
SPYPATCHFILE = BUILD / 'installers-conda.patch'

class BuildCondaPkg:
    """Base class for building a conda package for conda-based installer"""
    name = None
    norm = True
    source = None
    feedstock = None
    shallow_ver = None

    def __init__(self, data={}, debug=False, shallow=False):
        if False:
            while True:
                i = 10
        self.logger = getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.addHandler(h)
        self.logger.setLevel('INFO')
        self.debug = debug
        self._bld_src = BUILD / self.name
        self._fdstk_path = BUILD / self.feedstock.split('/')[-1]
        self._get_source(shallow=shallow)
        self._get_version()
        self._patch_source()
        self.data = {'version': self.version}
        self.data.update(data)
        self._recipe_patched = False

    def _get_source(self, shallow=False):
        if False:
            return 10
        'Clone source and feedstock to distribution directory for building'
        self._build_cleanup()
        if self.source == HERE.parent:
            self._bld_src = self.source
            self.repo = Repo(self.source)
        else:
            if self.source is not None:
                remote = self.source
                commit = 'HEAD'
            else:
                cfg = ConfigParser()
                cfg.read(EXTDEPS / self.name / '.gitrepo')
                remote = cfg['subrepo']['remote']
                commit = cfg['subrepo']['commit']
            kwargs = dict(to_path=self._bld_src)
            if shallow:
                kwargs.update(shallow_exclude=self.shallow_ver)
                self.logger.info(f'Cloning source shallow from tag {self.shallow_ver}...')
            else:
                self.logger.info('Cloning source...')
            self.repo = Repo.clone_from(remote, **kwargs)
            self.repo.git.checkout(commit)
        self.logger.info('Cloning feedstock...')
        Repo.clone_from(self.feedstock, to_path=self._fdstk_path)

    def _build_cleanup(self):
        if False:
            while True:
                i = 10
        'Remove cloned source and feedstock repositories'
        for src in [self._bld_src, self._fdstk_path]:
            if src.exists() and src != HERE.parent:
                logger.info(f'Removing {src}...')
                rmtree(src)

    def _get_version(self):
        if False:
            i = 10
            return i + 15
        'Get source version using setuptools_scm'
        v = get_version(self._bld_src, normalize=self.norm)
        self.version = v.lstrip('v').split('+')[0]

    def _patch_source(self):
        if False:
            print('Hello World!')
        pass

    def _patch_meta(self, meta):
        if False:
            print('Hello World!')
        return meta

    def _patch_build_script(self):
        if False:
            return 10
        pass

    def patch_recipe(self):
        if False:
            print('Hello World!')
        '\n        Patch conda build recipe\n\n        1. Patch meta.yaml\n        2. Patch build script\n        '
        if self._recipe_patched:
            return
        self.logger.info("Patching 'meta.yaml'...")
        file = self._fdstk_path / 'recipe' / 'meta.yaml'
        meta = file.read_text()
        for (k, v) in self.data.items():
            meta = re.sub(f'.*set {k} =.*', f'{{% set {k} = "{v}" %}}', meta)
        meta = re.sub('^(source:\\n)(  (url|sha256):.*\\n)*', f'\\g<1>  path: {self._bld_src.as_posix()}\\n', meta, flags=re.MULTILINE)
        meta = self._patch_meta(meta)
        file.rename(file.parent / ('_' + file.name))
        file.write_text(meta)
        self.logger.info(f"Patched 'meta.yaml' contents:\n{file.read_text()}")
        self._patch_build_script()
        self._recipe_patched = True

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build the conda package.\n\n        1. Patch the recipe\n        2. Build the package\n        3. Remove cloned repositories\n        '
        t0 = time()
        try:
            self.patch_recipe()
            self.logger.info(f'Building conda package {self.name}={self.version}...')
            check_call(['mamba', 'mambabuild', '--no-test', '--skip-existing', '--build-id-pat={n}', str(self._fdstk_path / 'recipe')])
        finally:
            self._recipe_patched = False
            if self.debug:
                self.logger.info('Keeping cloned source and feedstock')
            else:
                self._build_cleanup()
            elapse = timedelta(seconds=int(time() - t0))
            self.logger.info(f'Build time = {elapse}')

class SpyderCondaPkg(BuildCondaPkg):
    name = 'spyder'
    norm = False
    source = os.environ.get('SPYDER_SOURCE', HERE.parent)
    feedstock = 'https://github.com/conda-forge/spyder-feedstock'
    shallow_ver = 'v5.3.2'

    def _patch_source(self):
        if False:
            i = 10
            return i + 15
        self.logger.info('Creating Spyder menu file...')
        _menufile = RESOURCES / 'spyder-menu.json'
        self.menufile = BUILD / 'spyder-menu.json'
        (commit, branch) = self.repo.head.commit.name_rev.split()
        text = _menufile.read_text()
        text = text.replace('__PKG_VERSION__', self.version)
        text = text.replace('__SPY_BRANCH__', branch)
        text = text.replace('__SPY_COMMIT__', commit[:8])
        self.menufile.write_text(text)

    def _patch_meta(self, meta):
        if False:
            i = 10
            return i + 15
        meta = re.sub('^(build:\\n([ ]{2,}.*\\n)*)  osx_is_app:.*\\n', '\\g<1>', meta, flags=re.MULTILINE)
        meta = re.sub('^app:\\n(  .*\\n)+', '', meta, flags=re.MULTILINE)
        yaml = YAML()
        current_requirements = ['python']
        current_requirements += yaml.load(REQ_MAIN.read_text())['dependencies']
        if os.name == 'nt':
            win_requirements = yaml.load(REQ_WINDOWS.read_text())['dependencies']
            current_requirements += win_requirements
            current_requirements.append('ptyprocess >=0.5')
        elif sys.platform == 'darwin':
            mac_requirements = yaml.load(REQ_MAC.read_text())['dependencies']
            if 'python.app' in mac_requirements:
                mac_requirements.remove('python.app')
            current_requirements += mac_requirements
        else:
            linux_requirements = yaml.load(REQ_LINUX.read_text())['dependencies']
            current_requirements += linux_requirements
        cr_string = '\n    - '.join(current_requirements)
        meta = re.sub('^(requirements:\\n(.*\\n)+  run:\\n)(    .*\\n)+', f'\\g<1>    - {cr_string}\\n', meta, flags=re.MULTILINE)
        return meta

    def _patch_build_script(self):
        if False:
            i = 10
            return i + 15
        self.logger.info('Patching build script...')
        rel_menufile = self.menufile.relative_to(HERE.parent)
        if os.name == 'posix':
            logomark = 'branding/logo/logomark/spyder-logomark-background.png'
            file = self._fdstk_path / 'recipe' / 'build.sh'
            text = file.read_text()
            text += dedent(f'\n                # Create the Menu directory\n                mkdir -p "${{PREFIX}}/Menu"\n\n                # Copy menu.json template\n                cp "${{SRC_DIR}}/{rel_menufile}" "${{PREFIX}}/Menu/spyder-menu.json"\n\n                # Copy application icons\n                if [[ $OSTYPE == "darwin"* ]]; then\n                    cp "${{SRC_DIR}}/img_src/spyder.icns" "${{PREFIX}}/Menu/spyder.icns"\n                else\n                    cp "${{SRC_DIR}}/{logomark}" "${{PREFIX}}/Menu/spyder.png"\n                fi\n                ')
        if os.name == 'nt':
            file = self._fdstk_path / 'recipe' / 'bld.bat'
            text = file.read_text()
            text = text.replace('copy %RECIPE_DIR%\\menu-windows.json %MENU_DIR%\\spyder_shortcut.json', f'copy %SRC_DIR%\\{rel_menufile} %MENU_DIR%\\spyder-menu.json')
        file.rename(file.parent / ('_' + file.name))
        file.write_text(text)
        self.logger.info(f'Patched build script contents:\n{file.read_text()}')

class PylspCondaPkg(BuildCondaPkg):
    name = 'python-lsp-server'
    source = os.environ.get('PYTHON_LSP_SERVER_SOURCE')
    feedstock = 'https://github.com/conda-forge/python-lsp-server-feedstock'
    shallow_ver = 'v1.4.1'

class QtconsoleCondaPkg(BuildCondaPkg):
    name = 'qtconsole'
    source = os.environ.get('QTCONSOLE_SOURCE')
    feedstock = 'https://github.com/conda-forge/qtconsole-feedstock'
    shallow_ver = '5.3.1'

class SpyderKernelsCondaPkg(BuildCondaPkg):
    name = 'spyder-kernels'
    source = os.environ.get('SPYDER_KERNELS_SOURCE')
    feedstock = 'https://github.com/conda-forge/spyder-kernels-feedstock'
    shallow_ver = 'v2.3.1'
PKGS = {SpyderCondaPkg.name: SpyderCondaPkg, PylspCondaPkg.name: PylspCondaPkg, QtconsoleCondaPkg.name: QtconsoleCondaPkg, SpyderKernelsCondaPkg.name: SpyderKernelsCondaPkg}
if __name__ == '__main__':
    p = ArgumentParser(description=dedent('\n            Build conda packages to local channel.\n\n            This module builds conda packages for Spyder and external-deps for\n            inclusion in the conda-based installer. The following classes are\n            provided for each package:\n                SpyderCondaPkg\n                PylspCondaPkg\n                QdarkstyleCondaPkg\n                QtconsoleCondaPkg\n                SpyderKernelsCondaPkg\n\n            Spyder will be packaged from this repository (in its checked-out\n            state). qtconsole, spyder-kernels, and python-lsp-server will be\n            packaged from the remote and commit specified in their respective\n            .gitrepo files in external-deps.\n\n            Alternatively, any external-deps may be packaged from an arbitrary\n            git repository (in its checked out state) by setting the\n            appropriate environment variable from the following:\n                SPYDER_SOURCE\n                PYTHON_LSP_SERVER_SOURCE\n                QDARKSTYLE_SOURCE\n                QTCONSOLE_SOURCE\n                SPYDER_KERNELS_SOURCE\n            '), usage='python build_conda_pkgs.py [--build BUILD [BUILD] ...] [--debug] [--shallow]', formatter_class=RawTextHelpFormatter)
    p.add_argument('--debug', action='store_true', default=False, help='Do not remove cloned sources and feedstocks')
    p.add_argument('--build', nargs='+', default=PKGS.keys(), help=f'Space-separated list of packages to build. Default is {list(PKGS.keys())}')
    p.add_argument('--shallow', action='store_true', default=False, help='Perform shallow clone for build')
    args = p.parse_args()
    logger.info(f'Building local conda packages {list(args.build)}...')
    t0 = time()
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    if SPECS.exists():
        specs = yaml.load(SPECS.read_text())
    else:
        specs = {k: '' for k in PKGS}
    for k in args.build:
        pkg = PKGS[k](debug=args.debug, shallow=args.shallow)
        pkg.build()
        specs[k] = '=' + pkg.version
    yaml.dump(specs, SPECS)
    elapse = timedelta(seconds=int(time() - t0))
    logger.info(f'Total build time = {elapse}')