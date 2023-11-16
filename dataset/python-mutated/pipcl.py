"""
Python packaging operations, including PEP-517 support, for use by a `setup.py`
script.

The intention is to take care of as many packaging details as possible so that
setup.py contains only project-specific information, while also giving as much
flexibility as possible.

For example we provide a function `build_extension()` that can be used to build
a SWIG extension, but we also give access to the located compiler/linker so
that a `setup.py` script can take over the details itself.

Run doctests with: `python -m doctest pipcl.py`
"""
import base64
import glob
import hashlib
import inspect
import io
import os
import platform
import re
import shutil
import site
import setuptools
import subprocess
import sys
import sysconfig
import tarfile
import textwrap
import time
import zipfile
import wdev

class Package:
    '''
    Our constructor takes a definition of a Python package similar to that
    passed to `distutils.core.setup()` or `setuptools.setup()` (name, version,
    summary etc) plus callbacks for building, getting a list of sdist
    filenames, and cleaning.

    We provide methods that can be used to implement a Python package's
    `setup.py` supporting PEP-517.

    We also support basic command line handling for use
    with a legacy (pre-PEP-517) pip, as implemented
    by legacy distutils/setuptools and described in:
    https://pip.pypa.io/en/stable/reference/build-system/setup-py/

    Here is a `doctest` example of using pipcl to create a SWIG extension
    module. Requires `swig`.

    Create an empty test directory:

        >>> import os
        >>> import shutil
        >>> shutil.rmtree('pipcl_test', ignore_errors=1)
        >>> os.mkdir('pipcl_test')

    Create a `setup.py` which uses `pipcl` to define an extension module.

        >>> import textwrap
        >>> with open('pipcl_test/setup.py', 'w') as f:
        ...     _ = f.write(textwrap.dedent("""
        ...             import sys
        ...             import pipcl
        ...
        ...             def build():
        ...                 so_leaf = pipcl.build_extension(
        ...                         name = 'foo',
        ...                         path_i = 'foo.i',
        ...                         outdir = 'build',
        ...                         )
        ...                 return [
        ...                         ('build/foo.py', 'foo/__init__.py'),
        ...                         (f'build/{so_leaf}', f'foo/'),
        ...                         ('README', '$dist-info/'),
        ...                         ]
        ...
        ...             def sdist():
        ...                 return [
        ...                         'foo.i',
        ...                         'bar.i',
        ...                         'setup.py',
        ...                         'pipcl.py',
        ...                         'wdev.py',
        ...                         'README',
        ...                         ]
        ...
        ...             p = pipcl.Package(
        ...                     name = 'foo',
        ...                     version = '1.2.3',
        ...                     fn_build = build,
        ...                     fn_sdist = sdist,
        ...                     )
        ...
        ...             build_wheel = p.build_wheel
        ...             build_sdist = p.build_sdist
        ...
        ...             # Handle old-style setup.py command-line usage:
        ...             if __name__ == '__main__':
        ...                 p.handle_argv(sys.argv)
        ...             """))

    Create the files required by the above `setup.py` - the SWIG `.i` input
    file, the README file, and copies of `pipcl.py` and `wdev.py`.

        >>> with open('pipcl_test/foo.i', 'w') as f:
        ...     _ = f.write(textwrap.dedent("""
        ...             %include bar.i
        ...             %{
        ...             #include <stdio.h>
        ...             #include <string.h>
        ...             int bar(const char* text)
        ...             {
        ...                 printf("bar(): text: %s\\\\n", text);
        ...                 int len = (int) strlen(text);
        ...                 printf("bar(): len=%i\\\\n", len);
        ...                 fflush(stdout);
        ...                 return len;
        ...             }
        ...             %}
        ...             int bar(const char* text);
        ...             """))

        >>> with open('pipcl_test/bar.i', 'w') as f:
        ...     _ = f.write( '\\n')

        >>> with open('pipcl_test/README', 'w') as f:
        ...     _ = f.write(textwrap.dedent("""
        ...             This is Foo.
        ...             """))

        >>> root = os.path.dirname(__file__)
        >>> _ = shutil.copy2(f'{root}/pipcl.py', 'pipcl_test/pipcl.py')
        >>> _ = shutil.copy2(f'{root}/wdev.py', 'pipcl_test/wdev.py')

    Use `setup.py`'s command-line interface to build and install the extension
    module into root `pipcl_test/install`.

        >>> _ = subprocess.run(
        ...         f'cd pipcl_test && {sys.executable} setup.py --root install install',
        ...         shell=1, check=1)

    The actual install directory depends on `sysconfig.get_path('platlib')`:

        >>> if windows():
        ...     install_dir = 'pipcl_test/install'
        ... else:
        ...     install_dir = f'pipcl_test/install/{sysconfig.get_path("platlib").lstrip(os.sep)}'
        >>> assert os.path.isfile( f'{install_dir}/foo/__init__.py')

    Create a test script which asserts that Python function call `foo.bar(s)`
    returns the length of `s`, and run it with `PYTHONPATH` set to the install
    directory:

        >>> with open('pipcl_test/test.py', 'w') as f:
        ...     _ = f.write(textwrap.dedent("""
        ...             import sys
        ...             import foo
        ...             text = 'hello'
        ...             print(f'test.py: calling foo.bar() with text={text!r}')
        ...             sys.stdout.flush()
        ...             l = foo.bar(text)
        ...             print(f'test.py: foo.bar() returned: {l}')
        ...             assert l == len(text)
        ...             """))
        >>> r = subprocess.run(
        ...         f'{sys.executable} pipcl_test/test.py',
        ...         shell=1, check=1, text=1,
        ...         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ...         env=os.environ | dict(PYTHONPATH=install_dir),
        ...         )
        >>> print(r.stdout)
        test.py: calling foo.bar() with text='hello'
        bar(): text: hello
        bar(): len=5
        test.py: foo.bar() returned: 5
        <BLANKLINE>

    Check that building sdist and wheel succeeds. For now we don't attempt to
    check that the sdist and wheel actually work.

        >>> _ = subprocess.run(
        ...         f'cd pipcl_test && {sys.executable} setup.py sdist',
        ...         shell=1, check=1)

        >>> _ = subprocess.run(
        ...         f'cd pipcl_test && {sys.executable} setup.py bdist_wheel',
        ...         shell=1, check=1)

    Check that rebuild does nothing.

        >>> t0 = os.path.getmtime('pipcl_test/build/foo.py')
        >>> _ = subprocess.run(
        ...         f'cd pipcl_test && {sys.executable} setup.py bdist_wheel',
        ...         shell=1, check=1)
        >>> t = os.path.getmtime('pipcl_test/build/foo.py')
        >>> assert t == t0

    Check that touching bar.i forces rebuild.

        >>> os.utime('pipcl_test/bar.i')
        >>> _ = subprocess.run(
        ...         f'cd pipcl_test && {sys.executable} setup.py bdist_wheel',
        ...         shell=1, check=1)
        >>> t = os.path.getmtime('pipcl_test/build/foo.py')
        >>> assert t > t0

    Check that touching foo.i.cpp does not run swig, but does recompile/link.

        >>> t0 = time.time()
        >>> os.utime('pipcl_test/build/foo.i.cpp')
        >>> _ = subprocess.run(
        ...         f'cd pipcl_test && {sys.executable} setup.py bdist_wheel',
        ...         shell=1, check=1)
        >>> assert os.path.getmtime('pipcl_test/build/foo.py') <= t0
        >>> so = glob.glob('pipcl_test/build/*.so')
        >>> assert len(so) == 1
        >>> so = so[0]
        >>> assert os.path.getmtime(so) > t0

    Wheels and sdists

        Wheels:
            We generate wheels according to:
            https://packaging.python.org/specifications/binary-distribution-format/

            * `{name}-{version}.dist-info/RECORD` uses sha256 hashes.
            * We do not generate other `RECORD*` files such as
              `RECORD.jws` or `RECORD.p7s`.
            * `{name}-{version}.dist-info/WHEEL` has:

              * `Wheel-Version: 1.0`
              * `Root-Is-Purelib: false`
            * No support for signed wheels.

        Sdists:
            We generate sdist's according to:
            https://packaging.python.org/specifications/source-distribution-format/
    '''

    def __init__(self, name, version, platform=None, supported_platform=None, summary=None, description=None, description_content_type=None, keywords=None, home_page=None, download_url=None, author=None, author_email=None, maintainer=None, maintainer_email=None, license=None, classifier=None, requires_dist=None, requires_python=None, requires_external=None, project_url=None, provides_extra=None, root=None, fn_build=None, fn_clean=None, fn_sdist=None, tag_python=None, tag_abi=None, tag_platform=None, wheel_compression=zipfile.ZIP_DEFLATED, wheel_compresslevel=None):
        if False:
            print('Hello World!')
        "\n        The initial args before `root` define the package\n        metadata and closely follow the definitions in:\n        https://packaging.python.org/specifications/core-metadata/\n\n        Args:\n\n            name:\n                A string, the name of the Python package.\n            version:\n                A string, the version of the Python package. Also see PEP-440\n                `Version Identification and Dependency Specification`.\n            platform:\n                A string or list of strings.\n            supported_platform:\n                A string or list of strings.\n            summary:\n                A string, short description of the package.\n            description:\n                A string, a detailed description of the package.\n            description_content_type:\n                A string describing markup of `description` arg. For example\n                `text/markdown; variant=GFM`.\n            keywords:\n                A string containing comma-separated keywords.\n            home_page:\n                URL of home page.\n            download_url:\n                Where this version can be downloaded from.\n            author:\n                Author.\n            author_email:\n                Author email.\n            maintainer:\n                Maintainer.\n            maintainer_email:\n                Maintainer email.\n            license:\n                A string containing the license text. Written into metadata\n                file `COPYING`. Is also written into metadata itself if not\n                multi-line.\n            classifier:\n                A string or list of strings. Also see:\n\n                * https://pypi.org/pypi?%3Aaction=list_classifiers\n                * https://pypi.org/classifiers/\n\n            requires_dist:\n                A string or list of strings. Also see PEP-508.\n            requires_python:\n                A string or list of strings.\n            requires_external:\n                A string or list of strings.\n            project_url:\n                A string or list of strings, each of the form: `{name}, {url}`.\n            provides_extra:\n                A string or list of strings.\n\n            root:\n                Root of package, defaults to current directory.\n\n            fn_build:\n                A function taking no args, or a single `config_settings` dict\n                arg (as described in PEP-517), that builds the package.\n\n                Should return a list of items; each item should be a tuple of\n                two strings `(from_, to_)`, or a single string `path` which is\n                treated as the tuple `(path, path)`.\n\n                `from_` should be the path to a file; if a relative path it is\n                assumed to be relative to `root`.\n\n                `to_` identifies what the file should be called within a wheel\n                or when installing. If `to_` ends with `/`, the leaf of `from_`\n                is appended to it.\n\n                Initial `$dist-info/` in `_to` is replaced by\n                `{name}-{version}.dist-info/`; this is useful for license files\n                etc.\n\n                Initial `$data/` in `_to` is replaced by\n                `{name}-{version}.data/`. We do not enforce particular\n                subdirectories, instead it is up to `fn_build()` to specify\n                specific subdirectories such as `purelib`, `headers`,\n                `scripts`, `data` etc.\n\n                If we are building a wheel (e.g. `python setup.py bdist_wheel`,\n                or PEP-517 pip calls `self.build_wheel()`), we add file `from_`\n                to the wheel archive with name `to_`.\n\n                If we are installing (e.g. `install` command in\n                the argv passed to `self.handle_argv()`), then\n                we copy `from_` to `{sitepackages}/{to_}`, where\n                `sitepackages` is the installation directory, the\n                default being `sysconfig.get_path('platlib')` e.g.\n                `myvenv/lib/python3.9/site-packages/`.\n\n            fn_clean:\n                A function taking a single arg `all_` that cleans generated\n                files. `all_` is true iff `--all` is in argv.\n\n                For safety and convenience, can also returns a list of\n                files/directory paths to be deleted. Relative paths are\n                interpreted as relative to `root`. All paths are asserted to be\n                within `root`.\n\n            fn_sdist:\n                A function taking no args, or a single `config_settings` dict\n                arg (as described in PEP517), that returns a list of paths for\n                files that should be copied into the sdist. Each item in the\n                list can also be a tuple `(from_, to_)`, where `from_` is the\n                path of a file and `to_` is its name within the sdist.\n\n                Relative paths are interpreted as relative to `root`. It is an\n                error if a path does not exist or is not a file.\n\n                It can be convenient to use `pipcl.git_items()`.\n\n                The specification for sdists requires that the list contains\n                `pyproject.toml`; we enforce this with a diagnostic rather than\n                raising an exception, to allow legacy command-line usage.\n\n            tag_python:\n                First element of wheel tag defined in PEP-425. If None we use\n                `cp{version}`.\n\n                For example if code works with any Python version, one can use\n                'py3'.\n\n            tag_abi:\n                Second element of wheel tag defined in PEP-425. If None we use\n                `none`.\n\n            tag_platform:\n                Third element of wheel tag defined in PEP-425. Default is\n                `os.environ('AUDITWHEEL_PLAT')` if set, otherwise derived\n                from `setuptools.distutils.util.get_platform()` (was\n                `distutils.util.get_platform()` as specified in the PEP), e.g.\n                `openbsd_7_0_amd64`.\n\n                For pure python packages use: `tag_platform=any`\n\n            wheel_compression:\n                Used as `zipfile.ZipFile()`'s `compression` parameter when\n                creating wheels.\n\n            wheel_compresslevel:\n                Used as `zipfile.ZipFile()`'s `compresslevel` parameter when\n                creating wheels.\n\n        "
        assert name
        assert version

        def assert_str(v):
            if False:
                print('Hello World!')
            if v is not None:
                assert isinstance(v, str), f'Not a string: {v!r}'

        def assert_str_or_multi(v):
            if False:
                return 10
            if v is not None:
                assert isinstance(v, (str, tuple, list)), f'Not a string, tuple or list: {v!r}'
        assert_str(name)
        assert_str(version)
        assert_str_or_multi(platform)
        assert_str_or_multi(supported_platform)
        assert_str(summary)
        assert_str(description)
        assert_str(description_content_type)
        assert_str(keywords)
        assert_str(home_page)
        assert_str(download_url)
        assert_str(author)
        assert_str(author_email)
        assert_str(maintainer)
        assert_str(maintainer_email)
        assert_str(license)
        assert_str_or_multi(classifier)
        assert_str_or_multi(requires_dist)
        assert_str(requires_python)
        assert_str_or_multi(requires_external)
        assert_str_or_multi(project_url)
        assert_str_or_multi(provides_extra)
        assert re.match('([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$', name, re.IGNORECASE), f'Bad name: {name!r}'
        assert re.match('^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\\.post(0|[1-9][0-9]*))?(\\.dev(0|[1-9][0-9]*))?$', version), f'Bad version: {version!r}.'
        if tag_python:
            assert '-' not in tag_python
        if tag_abi:
            assert '-' not in tag_abi
        if tag_platform:
            assert '-' not in tag_platform
        self.name = name
        self.version = version
        self.platform = platform
        self.supported_platform = supported_platform
        self.summary = summary
        self.description = description
        self.description_content_type = description_content_type
        self.keywords = keywords
        self.home_page = home_page
        self.download_url = download_url
        self.author = author
        self.author_email = author_email
        self.maintainer = maintainer
        self.maintainer_email = maintainer_email
        self.license = license
        self.classifier = classifier
        self.requires_dist = requires_dist
        self.requires_python = requires_python
        self.requires_external = requires_external
        self.project_url = project_url
        self.provides_extra = provides_extra
        self.root = os.path.abspath(root if root else os.getcwd())
        self.fn_build = fn_build
        self.fn_clean = fn_clean
        self.fn_sdist = fn_sdist
        self.tag_python = tag_python
        self.tag_abi = tag_abi
        self.tag_platform = tag_platform
        self.wheel_compression = wheel_compression
        self.wheel_compresslevel = wheel_compresslevel

    def build_wheel(self, wheel_directory, config_settings=None, metadata_directory=None):
        if False:
            i = 10
            return i + 15
        '\n        A PEP-517 `build_wheel()` function.\n\n        Also called by `handle_argv()` to handle the `bdist_wheel` command.\n\n        Returns leafname of generated wheel within `wheel_directory`.\n        '
        log2(f' wheel_directory={wheel_directory!r} config_settings={config_settings!r} metadata_directory={metadata_directory!r}')
        if self.tag_python:
            tag_python = self.tag_python
        else:
            tag_python = 'cp' + ''.join(platform.python_version().split('.')[:2])
        if self.tag_abi:
            tag_abi = self.tag_abi
        else:
            tag_abi = 'none'
        tag_platform = None
        if not tag_platform:
            tag_platform = self.tag_platform
        if not tag_platform:
            tag_platform = os.environ.get('AUDITWHEEL_PLAT')
        if not tag_platform:
            tag_platform = setuptools.distutils.util.get_platform().replace('-', '_').replace('.', '_')
            m = re.match('^(macosx_[0-9]+)(_[^0-9].+)$', tag_platform)
            if m:
                tag_platform2 = f'{m.group(1)}_0{m.group(2)}'
                log2(f'Changing from {tag_platform!r} to {tag_platform2!r}')
                tag_platform = tag_platform2
        tag = f'{tag_python}-{tag_abi}-{tag_platform}'
        path = f'{wheel_directory}/{self.name}-{self.version}-{tag}.whl'
        items = list()
        if self.fn_build:
            items = self._call_fn_build(config_settings)
        log2(f'Creating wheel: {path}')
        os.makedirs(wheel_directory, exist_ok=True)
        record = _Record()
        with zipfile.ZipFile(path, 'w', self.wheel_compression, self.wheel_compresslevel) as z:

            def add_file(from_, to_):
                if False:
                    while True:
                        i = 10
                z.write(from_, to_)
                record.add_file(from_, to_)

            def add_str(content, to_):
                if False:
                    return 10
                z.writestr(to_, content)
                record.add_content(content, to_)
            dist_info_dir = self._dist_info_dir()
            for item in items:
                ((from_abs, from_rel), (to_abs, to_rel)) = self._fromto(item)
                add_file(from_abs, to_rel)
            add_str(f'Wheel-Version: 1.0\nGenerator: pipcl\nRoot-Is-Purelib: false\nTag: {tag}\n', f'{dist_info_dir}/WHEEL')
            add_str(self._metainfo(), f'{dist_info_dir}/METADATA')
            if self.license:
                add_str(self.license, f'{dist_info_dir}/COPYING')
            z.writestr(f'{dist_info_dir}/RECORD', record.get(f'{dist_info_dir}/RECORD'))
        st = os.stat(path)
        log1(f'Have created wheel size={st.st_size}: {path}')
        if g_verbose >= 2:
            with zipfile.ZipFile(path, compression=self.wheel_compression) as z:
                log2(f'Contents are:')
                for zi in sorted(z.infolist(), key=lambda z: z.filename):
                    log2(f'    {zi.file_size: 10d} {zi.filename}')
        return os.path.basename(path)

    def build_sdist(self, sdist_directory, formats, config_settings=None):
        if False:
            print('Hello World!')
        '\n        A PEP-517 `build_sdist()` function.\n\n        Also called by `handle_argv()` to handle the `sdist` command.\n\n        Returns leafname of generated archive within `sdist_directory`.\n        '
        log2(f' sdist_directory={sdist_directory!r} formats={formats!r} config_settings={config_settings!r}')
        if formats and formats != 'gztar':
            raise Exception(f'Unsupported: formats={formats}')
        items = list()
        if self.fn_sdist:
            if inspect.signature(self.fn_sdist).parameters:
                items = self.fn_sdist(config_settings)
            else:
                items = self.fn_sdist()
        manifest = []
        names_in_tar = []

        def check_name(name):
            if False:
                for i in range(10):
                    print('nop')
            if name in names_in_tar:
                raise Exception(f'Name specified twice: {name}')
            names_in_tar.append(name)
        prefix = f'{self.name}-{self.version}'

        def add_content(tar, name, contents):
            if False:
                i = 10
                return i + 15
            '\n            Adds item called `name` to `tarfile.TarInfo` `tar`, containing\n            `contents`. If contents is a string, it is encoded using utf8.\n            '
            log2(f'Adding: {name}')
            if isinstance(contents, str):
                contents = contents.encode('utf8')
            check_name(name)
            ti = tarfile.TarInfo(f'{prefix}/{name}')
            ti.size = len(contents)
            ti.mtime = time.time()
            tar.addfile(ti, io.BytesIO(contents))

        def add_file(tar, path_abs, name):
            if False:
                for i in range(10):
                    print('nop')
            log2(f'Adding file: {os.path.relpath(path_abs)} => {name}')
            check_name(name)
            tar.add(path_abs, f'{prefix}/{name}', recursive=False)
        os.makedirs(sdist_directory, exist_ok=True)
        tarpath = f'{sdist_directory}/{prefix}.tar.gz'
        log2(f'Creating sdist: {tarpath}')
        with tarfile.open(tarpath, 'w:gz') as tar:
            found_pyproject_toml = False
            for item in items:
                ((from_abs, from_rel), (to_abs, to_rel)) = self._fromto(item)
                if from_abs.startswith(f'{os.path.abspath(sdist_directory)}/'):
                    assert 0, f'Path is inside sdist_directory={sdist_directory}: {from_abs!r}'
                assert os.path.exists(from_abs), f'Path does not exist: {from_abs!r}'
                assert os.path.isfile(from_abs), f'Path is not a file: {from_abs!r}'
                if to_rel == 'pyproject.toml':
                    found_pyproject_toml = True
                add_file(tar, from_abs, to_rel)
                manifest.append(to_rel)
            if not found_pyproject_toml:
                log0(f'Warning: no pyproject.toml specified.')
            add_content(tar, f'PKG-INFO', self._metainfo())
            if self.license:
                if 'COPYING' in names_in_tar:
                    log2(f'Not writing .license because file already in sdist: COPYING')
                else:
                    add_content(tar, f'COPYING', self.license)
        log1(f'Have created sdist: {tarpath}')
        return os.path.basename(tarpath)

    def _call_fn_build(self, config_settings=None):
        if False:
            return 10
        assert self.fn_build
        log2(f'calling self.fn_build={self.fn_build}')
        if inspect.signature(self.fn_build).parameters:
            ret = self.fn_build(config_settings)
        else:
            ret = self.fn_build()
        assert isinstance(ret, (list, tuple)), f'Expected list/tuple from {self.fn_build} but got: {ret!r}'
        return ret

    def _argv_clean(self, all_):
        if False:
            return 10
        '\n        Called by `handle_argv()`.\n        '
        if not self.fn_clean:
            return
        paths = self.fn_clean(all_)
        if paths:
            if isinstance(paths, str):
                paths = (paths,)
            for path in paths:
                if not os.path.isabs(path):
                    path = ps.path.join(self.root, path)
                path = os.path.abspath(path)
                assert path.startswith(self.root + os.sep), f'path={path!r} does not start with root={self.root + os.sep!r}'
                log2(f'Removing: {path}')
                shutil.rmtree(path, ignore_errors=True)

    def install(self, record_path=None, root=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called by `handle_argv()` to handle `install` command..\n        '
        log2(f'record_path={record_path!r} root={root!r}')
        items = list()
        if self.fn_build:
            items = self._call_fn_build(dict())
        root2 = install_dir(root)
        log2(f'root2={root2!r}')
        log1(f'Installing into: {root2!r}')
        dist_info_dir = self._dist_info_dir()
        if not record_path:
            record_path = f'{root2}/{dist_info_dir}/RECORD'
        record = _Record()

        def add_file(from_abs, from_rel, to_abs, to_rel):
            if False:
                while True:
                    i = 10
            log2(f'Copying from {from_rel} to {to_abs}')
            os.makedirs(os.path.dirname(to_abs), exist_ok=True)
            shutil.copy2(from_abs, to_abs)
            record.add_file(from_rel, to_rel)

        def add_str(content, to_abs, to_rel):
            if False:
                i = 10
                return i + 15
            log2(f'Writing to: {to_abs}')
            os.makedirs(os.path.dirname(to_abs), exist_ok=True)
            with open(to_abs, 'w') as f:
                f.write(content)
            record.add_content(content, to_rel)
        for item in items:
            ((from_abs, from_rel), (to_abs, to_rel)) = self._fromto(item)
            to_abs2 = f'{root2}/{to_rel}'
            add_file(from_abs, from_rel, to_abs2, to_rel)
        add_str(self._metainfo(), f'{root2}/{dist_info_dir}/METADATA', f'{dist_info_dir}/METADATA')
        log2(f'Writing to: {record_path}')
        with open(record_path, 'w') as f:
            f.write(record.get())
        log2(f'Finished.')

    def _argv_dist_info(self, root):
        if False:
            while True:
                i = 10
        "\n        Called by `handle_argv()`. There doesn't seem to be any documentation\n        for `setup.py dist_info`, but it appears to be like `egg_info` except\n        it writes to a slightly different directory.\n        "
        if root is None:
            root = f'{self.name}-{self.version}.dist-info'
        self._write_info(f'{root}/METADATA')
        if self.license:
            with open(f'{root}/COPYING', 'w') as f:
                f.write(self.license)

    def _argv_egg_info(self, egg_base):
        if False:
            while True:
                i = 10
        '\n        Called by `handle_argv()`.\n        '
        if egg_base is None:
            egg_base = '.'
        self._write_info(f'{egg_base}/.egg-info')

    def _write_info(self, dirpath=None):
        if False:
            while True:
                i = 10
        '\n        Writes egg/dist info to files in directory `dirpath` or `self.root` if\n        `None`.\n        '
        if dirpath is None:
            dirpath = self.root
        log2(f'Creating files in directory {dirpath}')
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, 'PKG-INFO'), 'w') as f:
            f.write(self._metainfo())

    def handle_argv(self, argv):
        if False:
            return 10
        '\n        Attempt to handles old-style (pre PEP-517) command line passed by\n        old releases of pip to a `setup.py` script, and manual running of\n        `setup.py`.\n\n        This is partial support at best.\n        '
        global g_verbose

        class ArgsRaise:
            pass

        class Args:
            """
            Iterates over argv items.
            """

            def __init__(self, argv):
                if False:
                    while True:
                        i = 10
                self.items = iter(argv)

            def next(self, eof=ArgsRaise):
                if False:
                    print('Hello World!')
                '\n                Returns next arg. If no more args, we return <eof> or raise an\n                exception if <eof> is ArgsRaise.\n                '
                try:
                    return next(self.items)
                except StopIteration:
                    if eof is ArgsRaise:
                        raise Exception('Not enough args')
                    return eof
        command = None
        opt_all = None
        opt_dist_dir = 'dist'
        opt_egg_base = None
        opt_formats = None
        opt_install_headers = None
        opt_record = None
        opt_root = None
        args = Args(argv[1:])
        while 1:
            arg = args.next(None)
            if arg is None:
                break
            elif arg in ('-h', '--help', '--help-commands'):
                log0(textwrap.dedent('\n                        Usage:\n                            [<options>...] <command> [<options>...]\n                        Commands:\n                            bdist_wheel\n                                Creates a wheel called\n                                <dist-dir>/<name>-<version>-<details>.whl, where\n                                <dist-dir> is "dist" or as specified by --dist-dir,\n                                and <details> encodes ABI and platform etc.\n                            clean\n                                Cleans build files.\n                            dist_info\n                                Creates files in <name>-<version>.dist-info/ or\n                                directory specified by --egg-base.\n                            egg_info\n                                Creates files in .egg-info/ or directory\n                                directory specified by --egg-base.\n                            install\n                                Builds and installs. Writes installation\n                                information to <record> if --record was\n                                specified.\n                            sdist\n                                Make a source distribution:\n                                    <dist-dir>/<name>-<version>.tar.gz\n                        Options:\n                            --all\n                                Used by "clean".\n                            --compile\n                                Ignored.\n                            --dist-dir | -d <dist-dir>\n                                Default is "dist".\n                            --egg-base <egg-base>\n                                Used by "egg_info".\n                            --formats <formats>\n                                Used by "sdist".\n                            --install-headers <directory>\n                                Ignored.\n                            --python-tag <python-tag>\n                                Ignored.\n                            --record <record>\n                                Used by "install".\n                            --root <path>\n                                Used by "install".\n                            --single-version-externally-managed\n                                Ignored.\n                            --verbose -v\n                                Extra diagnostics.\n                        Other:\n                            windows-vs [-y <year>] [-v <version>] [-g <grade] [--verbose]\n                                Windows only; looks for matching Visual Studio.\n                            windows-python [-v <version>] [--verbose]\n                                Windows only; looks for matching Python.\n                        '))
                return
            elif arg in ('bdist_wheel', 'clean', 'dist_info', 'egg_info', 'install', 'sdist'):
                assert command is None, 'Two commands specified: {command} and {arg}.'
                command = arg
            elif arg == '--all':
                opt_all = True
            elif arg == '--compile':
                pass
            elif arg == '--dist-dir' or arg == '-d':
                opt_dist_dir = args.next()
            elif arg == '--egg-base':
                opt_egg_base = args.next()
            elif arg == '--formats':
                opt_formats = args.next()
            elif arg == '--install-headers':
                opt_install_headers = args.next()
            elif arg == '--python-tag':
                pass
            elif arg == '--record':
                opt_record = args.next()
            elif arg == '--root':
                opt_root = args.next()
            elif arg == '--single-version-externally-managed':
                pass
            elif arg == '--verbose' or arg == '-v':
                g_verbose += 1
            elif arg == 'windows-vs':
                command = arg
                break
            elif arg == 'windows-python':
                command = arg
                break
            else:
                raise Exception(f'Unrecognised arg: {arg}')
        assert command, 'No command specified'
        log1(f'Handling command={command}')
        if 0:
            pass
        elif command == 'bdist_wheel':
            self.build_wheel(opt_dist_dir)
        elif command == 'clean':
            self._argv_clean(opt_all)
        elif command == 'dist_info':
            self._argv_dist_info(opt_egg_base)
        elif command == 'egg_info':
            self._argv_egg_info(opt_egg_base)
        elif command == 'install':
            self.install(opt_record, opt_root)
        elif command == 'sdist':
            self.build_sdist(opt_dist_dir, opt_formats)
        elif command == 'windows-python':
            version = None
            while 1:
                arg = args.next(None)
                if arg is None:
                    break
                elif arg == '-v':
                    version = args.next()
                elif arg == '--verbose':
                    g_verbose += 1
                else:
                    assert 0, f'Unrecognised arg={arg!r}'
            python = wdev.WindowsPython(version=version)
            print(f"Python is:\n{python.description_ml('    ')}")
        elif command == 'windows-vs':
            grade = None
            version = None
            year = None
            while 1:
                arg = args.next(None)
                if arg is None:
                    break
                elif arg == '-g':
                    grade = args.next()
                elif arg == '-v':
                    version = args.next()
                elif arg == '-y':
                    year = args.next()
                elif arg == '--verbose':
                    g_verbose += 1
                else:
                    assert 0, f'Unrecognised arg={arg!r}'
            vs = wdev.WindowsVS(year=year, grade=grade, version=version)
            print(f"Visual Studio is:\n{vs.description_ml('    ')}")
        else:
            assert 0, f'Unrecognised command: {command}'
        log2(f'Finished handling command: {command}')

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'{{name={self.name!r} version={self.version!r} platform={self.platform!r} supported_platform={self.supported_platform!r} summary={self.summary!r} description={self.description!r} description_content_type={self.description_content_type!r} keywords={self.keywords!r} home_page={self.home_page!r} download_url={self.download_url!r} author={self.author!r} author_email={self.author_email!r} maintainer={self.maintainer!r} maintainer_email={self.maintainer_email!r} license={self.license!r} classifier={self.classifier!r} requires_dist={self.requires_dist!r} requires_python={self.requires_python!r} requires_external={self.requires_external!r} project_url={self.project_url!r} provides_extra={self.provides_extra!r} root={self.root!r} fn_build={self.fn_build!r} fn_sdist={self.fn_sdist!r} fn_clean={self.fn_clean!r} tag_python={self.tag_python!r} tag_abi={self.tag_abi!r} tag_platform={self.tag_platform!r}}}'

    def _dist_info_dir(self):
        if False:
            while True:
                i = 10
        return f'{self.name}-{self.version}.dist-info'

    def _metainfo(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns text for `.egg-info/PKG-INFO` file, or `PKG-INFO` in an sdist\n        `.tar.gz` file, or `...dist-info/METADATA` in a wheel.\n        '
        ret = ['']

        def add(key, value):
            if False:
                for i in range(10):
                    print('nop')
            if value is None:
                return
            if isinstance(value, (tuple, list)):
                for v in value:
                    add(key, v)
                return
            if key == 'License' and '\n' in value:
                log1(f'Omitting license because contains newline(s).')
                return
            assert '\n' not in value, f'key={key} value contains newline: {value!r}'
            if key == 'Project-URL':
                assert value.count(',') == 1, f'For key={key!r}, should have one comma in {value!r}.'
            ret[0] += f'{key}: {value}\n'
        add('Metadata-Version', '2.1')
        for name in ('Name', 'Version', 'Platform', 'Supported-Platform', 'Summary', 'Description-Content-Type', 'Keywords', 'Home-page', 'Download-URL', 'Author', 'Author-email', 'Maintainer', 'Maintainer-email', 'License', 'Classifier', 'Requires-Dist', 'Requires-Python', 'Requires-External', 'Project-URL', 'Provides-Extra'):
            identifier = name.lower().replace('-', '_')
            add(name, getattr(self, identifier))
        ret = ret[0]
        if self.description:
            ret += '\n'
            ret += self.description.strip()
            ret += '\n'
        return ret

    def _path_relative_to_root(self, path, assert_within_root=True):
        if False:
            return 10
        '\n        Returns `(path_abs, path_rel)`, where `path_abs` is absolute path and\n        `path_rel` is relative to `self.root`.\n\n        Interprets `path` as relative to `self.root` if not absolute.\n\n        We use `os.path.realpath()` to resolve any links.\n\n        if `assert_within_root` is true, assert-fails if `path` is not within\n        `self.root`.\n        '
        if os.path.isabs(path):
            p = path
        else:
            p = os.path.join(self.root, path)
        p = os.path.realpath(os.path.abspath(p))
        if assert_within_root:
            assert p.startswith(self.root + os.sep) or p == self.root, f'Path not within root={self.root + os.sep!r}: path={path!r} p={p!r}'
        p_rel = os.path.relpath(p, self.root)
        return (p, p_rel)

    def _fromto(self, p):
        if False:
            return 10
        '\n        Returns `((from_abs, from_rel), (to_abs, to_rel))`.\n\n        If `p` is a string we convert to `(p, p)`. Otherwise we assert\n        that `p` is a tuple of two string, `(from_, to_)`. Non-absolute\n        paths are assumed to be relative to `self.root`. If `to_` is\n        empty or ends with `/`, we append the leaf of `from_`.\n\n        If `to_` starts with `$dist-info/`, we replace this with\n        `self._dist_info_dir()`.\n\n        If `to_` starts with `$data/`, we replace this with\n        `{self.name}-{self.version}.data/`.\n\n        `from_abs` and `to_abs` are absolute paths. We assert that `to_abs` is\n        `within self.root`.\n\n        `from_rel` and `to_rel` are derived from the `_abs` paths and are\n        `relative to self.root`.\n        '
        ret = None
        if isinstance(p, str):
            ret = (p, p)
        elif isinstance(p, tuple) and len(p) == 2:
            (from_, to_) = p
            if isinstance(from_, str) and isinstance(to_, str):
                ret = (from_, to_)
        assert ret, 'p should be str or (str, str), but is: {p}'
        (from_, to_) = ret
        if to_.endswith('/') or to_ == '':
            to_ += os.path.basename(from_)
        prefix = '$dist-info/'
        if to_.startswith(prefix):
            to_ = f'{self._dist_info_dir()}/{to_[len(prefix):]}'
        prefix = '$data/'
        if to_.startswith(prefix):
            to_ = f'{self.name}-{self.version}.data/{to_[len(prefix):]}'
        from_ = self._path_relative_to_root(from_, assert_within_root=False)
        to_ = self._path_relative_to_root(to_)
        return (from_, to_)

def build_extension(name, path_i, outdir, builddir=None, includes=None, defines=None, libpaths=None, libs=None, optimise=True, debug=False, compiler_extra='', linker_extra='', swig='swig', cpp=True, prerequisites_swig=None, prerequisites_compile=None, prerequisites_link=None, infer_swig_includes=True):
    if False:
        print('Hello World!')
    "\n    Builds a Python extension module using SWIG. Works on Windows, Linux, MacOS\n    and OpenBSD.\n\n    On Unix, sets rpath when linking shared libraries.\n\n    Args:\n        name:\n            Name of generated extension module.\n        path_i:\n            Path of input SWIG `.i` file. Internally we use swig to generate a\n            corresponding `.c` or `.cpp` file.\n        outdir:\n            Output directory for generated files:\n\n                * `{outdir}/{name}.py`\n                * `{outdir}/_{name}.so`     # Unix\n                * `{outdir}/_{name}.*.pyd`  # Windows\n            We return the leafname of the `.so` or `.pyd` file.\n        builddir:\n            Where to put intermediate files, for example the .cpp file\n            generated by swig and `.d` dependency files. Default is `outdir`.\n        includes:\n            A string, or a sequence of extra include directories to be prefixed\n            with `-I`.\n        defines:\n            A string, or a sequence of extra preprocessor defines to be\n            prefixed with `-D`.\n        libpaths\n            A string, or a sequence of library paths to be prefixed with\n            `/LIBPATH:` on Windows or `-L` on Unix.\n        libs\n            A string, or a sequence of library names to be prefixed with `-l`.\n        optimise:\n            Whether to use compiler optimisations.\n        debug:\n            Whether to build with debug symbols.\n        compiler_extra:\n            Extra compiler flags.\n        linker_extra:\n            Extra linker flags.\n        swig:\n            Base swig command.\n        cpp:\n            If true we tell SWIG to generate C++ code instead of C.\n        prerequisites_swig:\n        prerequisites_compile:\n        prerequisites_link:\n\n            [These are mainly for use on Windows. On other systems we\n            automatically generate dynamic dependencies using swig/compile/link\n            commands' `-MD` and `-MF` args.]\n\n            Sequences of extra input files/directories that should force\n            running of swig, compile or link commands if they are newer than\n            any existing generated SWIG `.i` file, compiled object file or\n            shared library file.\n\n            If present, the first occurrence of `True` or `False` forces re-run\n            or no re-run. Any occurrence of None is ignored. If an item is a\n            directory path we look for newest file within the directory tree.\n\n            If not a sequence, we convert into a single-item list.\n\n            prerequisites_swig\n\n                We use swig's -MD and -MF args to generate dynamic dependencies\n                automatically, so this is not usually required.\n\n            prerequisites_compile\n            prerequisites_link\n\n                On non-Windows we use cc's -MF and -MF args to generate dynamic\n                dependencies so this is not usually required.\n        infer_swig_includes:\n            If true, we extract `-I<path>` and `-I <path>` args from\n            `compile_extra` (also `/I` on windows) and use them with swig so\n            that it can see the same header files as C/C++. This is useful\n            when using enviromment variables such as `CC` and `CXX` to set\n            `compile_extra.\n\n    Returns the leafname of the generated library file within `outdir`, e.g.\n    `_{name}.so` on Unix or `_{name}.cp311-win_amd64.pyd` on Windows.\n    "
    if builddir is None:
        builddir = outdir
    includes_text = _flags(includes, '-I')
    defines_text = _flags(defines, '-D')
    libpaths_text = _flags(libpaths, '/LIBPATH:', '"') if windows() else _flags(libpaths, '-L')
    libs_text = _flags(libs, '-l')
    path_cpp = f'{builddir}/{os.path.basename(path_i)}'
    path_cpp += '.cpp' if cpp else '.c'
    os.makedirs(outdir, exist_ok=True)
    if infer_swig_includes:
        swig_includes_extra = ''
        compiler_extra_items = compiler_extra.split()
        i = 0
        while i < len(compiler_extra_items):
            item = compiler_extra_items[i]
            if item == '-I' or (windows() and item == '/I'):
                swig_includes_extra += f' -I{compiler_extra_items[i + 1]}'
                i += 1
            elif item.startswith('-I') or (windows() and item.startswith('/I')):
                swig_includes_extra += f' -I{compiler_extra_items[i][2:]}'
            i += 1
        swig_includes_extra = swig_includes_extra.strip()
    deps_path = f'{path_cpp}.d'
    prerequisites_swig2 = _get_prerequisites(deps_path)
    run_if(f"\n            {swig}\n                -Wall\n                {('-c++' if cpp else '')}\n                -python\n                -module {name}\n                -outdir {outdir}\n                -o {path_cpp}\n                -MD -MF {deps_path}\n                {includes_text}\n                {swig_includes_extra}\n                {path_i}\n            ", path_cpp, path_i, prerequisites_swig, prerequisites_swig2)
    path_so_leaf = f'_{name}{_so_suffix()}'
    path_so = f'{outdir}/{path_so_leaf}'
    if windows():
        path_obj = f'{path_so}.obj'
        permissive = '/permissive-'
        EHsc = '/EHsc'
        T = '/Tp' if cpp else '/Tc'
        optimise2 = '/DNDEBUG /O2' if optimise else '/D_DEBUG'
        debug2 = ''
        if debug:
            debug2 = '/Zi'
        (command, pythonflags) = base_compiler(cpp=cpp)
        command = f"""\n                {command}\n                    # General:\n                    /c                          # Compiles without linking.\n                    {EHsc}                      # Enable "Standard C++ exception handling".\n\n                    #/MD                         # Creates a multithreaded DLL using MSVCRT.lib.\n                    {('/MDd' if debug else '/MD')}\n\n                    # Input/output files:\n                    {T}{path_cpp}               # /Tp specifies C++ source file.\n                    /Fo{path_obj}               # Output file.\n\n                    # Include paths:\n                    {includes_text}\n                    {pythonflags.includes}      # Include path for Python headers.\n\n                    # Code generation:\n                    {optimise2}\n                    {debug2}\n                    {permissive}                # Set standard-conformance mode.\n\n                    # Diagnostics:\n                    #/FC                         # Display full path of source code files passed to cl.exe in diagnostic text.\n                    /W3                         # Sets which warning level to output. /W3 is IDE default.\n                    /diagnostics:caret          # Controls the format of diagnostic messages.\n                    /nologo                     #\n\n                    {defines_text}\n                    {compiler_extra}\n                """
        run_if(command, path_obj, path_cpp, prerequisites_compile)
        (command, pythonflags) = base_linker(cpp=cpp)
        debug2 = '/DEBUG' if debug else ''
        (base, _) = os.path.splitext(path_so_leaf)
        command = f'\n                {command}\n                    /DLL                    # Builds a DLL.\n                    /EXPORT:PyInit__{name}  # Exports a function.\n                    /IMPLIB:{base}.lib      # Overrides the default import library name.\n                    {libpaths_text}\n                    {pythonflags.ldflags}\n                    /OUT:{path_so}          # Specifies the output file name.\n                    {debug2}\n                    /nologo\n                    {libs_text}\n                    {path_obj}\n                    {linker_extra}\n                '
        run_if(command, path_so, path_obj, prerequisites_link)
    else:
        (command, pythonflags) = base_compiler(cpp=cpp)
        general_flags = ''
        if debug:
            general_flags += ' -g'
        if optimise:
            general_flags += ' -O2 -DNDEBUG'
        if darwin():
            rpath_flag = '-Wl,-rpath,@loader_path/'
            general_flags += ' -undefined dynamic_lookup'
        elif pyodide():
            log0(f'pyodide: PEP-3149 suffix untested, so omitting. _so_suffix()={_so_suffix()!r}.')
            path_so_leaf = f'_{name}.so'
            path_so = f'{outdir}/{path_so_leaf}'
            rpath_flag = ''
        else:
            rpath_flag = "-Wl,-rpath,'$ORIGIN',-z,origin"
        path_so = f'{outdir}/{path_so_leaf}'
        prerequisites = list()
        if pyodide():
            prerequisites_compile_path = f'{path_cpp}.o.d'
            prerequisites += _get_prerequisites(prerequisites_compile_path)
            command = f'\n                    {command}\n                        -fPIC\n                        {general_flags.strip()}\n                        {pythonflags.includes}\n                        {includes_text}\n                        {defines_text}\n                        -MD -MF {prerequisites_compile_path}\n                        -c {path_cpp}\n                        -o {path_cpp}.o\n                        {compiler_extra}\n                    '
            prerequisites_link_path = f'{path_cpp}.o.d'
            prerequisites += _get_prerequisites(prerequisites_link_path)
            (ld, _) = base_linker(cpp=cpp)
            command += f'\n                    && {ld}\n                        {path_cpp}.o\n                        -o {path_so}\n                        -MD -MF {prerequisites_link_path}\n                        {rpath_flag}\n                        {libpaths_text}\n                        {libs_text}\n                        {linker_extra}\n                        {pythonflags.ldflags}\n                    '
        else:
            prerequisites_path = f'{path_so}.d'
            prerequisites = _get_prerequisites(prerequisites_path)
            command = f'\n                    {command}\n                        -fPIC\n                        -shared\n                        {general_flags.strip()}\n                        {pythonflags.includes}\n                        {includes_text}\n                        {defines_text}\n                        {path_cpp}\n                        -MD -MF {prerequisites_path}\n                        -o {path_so}\n                        {compiler_extra}\n                        {libpaths_text}\n                        {linker_extra}\n                        {pythonflags.ldflags}\n                        {libs_text}\n                        {rpath_flag}\n                    '
        command_was_run = run_if(command, path_so, path_cpp, prerequisites_compile, prerequisites_link, prerequisites)
        if command_was_run and darwin():
            sublibraries = list()
            for lib in libs:
                for libpath in libpaths:
                    found = list()
                    for suffix in ('.so', '.dylib'):
                        path = f'{libpath}/lib{os.path.basename(lib)}{suffix}'
                        if os.path.exists(path):
                            found.append(path)
                    if found:
                        assert len(found) == 1, f'More than one file matches lib={lib!r}: {found}'
                        sublibraries.append(found[0])
                        break
                else:
                    log2(f'Warning: can not find path of lib={lib!r} in libpaths={libpaths}')
            macos_patch(path_so, *sublibraries)
    return path_so_leaf

def base_compiler(vs=None, pythonflags=None, cpp=False, use_env=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns basic compiler command and PythonFlags.\n\n    Args:\n        vs:\n            Windows only. A `wdev.WindowsVS` instance or None to use default\n            `wdev.WindowsVS` instance.\n        pythonflags:\n            A `pipcl.PythonFlags` instance or None to use default\n            `pipcl.PythonFlags` instance.\n        cpp:\n            If true we return C++ compiler command instead of C. On Windows\n            this has no effect - we always return `cl.exe`.\n        use_env:\n            If true we return '$CC' or '$CXX' if the corresponding\n            environmental variable is set (without evaluating with `getenv()`\n            or `os.environ`).\n\n    Returns `(cc, pythonflags)`:\n        cc:\n            C or C++ command. On Windows this is of the form\n            `{vs.vcvars}&&{vs.cl}`; otherwise it is typically `cc` or `c++`.\n        pythonflags:\n            The `pythonflags` arg or a new `pipcl.PythonFlags` instance.\n    "
    if not pythonflags:
        pythonflags = PythonFlags()
    cc = None
    if use_env:
        if cpp:
            if os.environ.get('CXX'):
                cc = '$CXX'
        elif os.environ.get('CC'):
            cc = '$CC'
    if cc:
        pass
    elif windows():
        if not vs:
            vs = wdev.WindowsVS()
        cc = f'"{vs.vcvars}"&&"{vs.cl}"'
    elif wasm():
        cc = 'em++' if cpp else 'emcc'
    else:
        cc = 'c++' if cpp else 'cc'
    cc = macos_add_cross_flags(cc)
    return (cc, pythonflags)

def base_linker(vs=None, pythonflags=None, cpp=False, use_env=True):
    if False:
        i = 10
        return i + 15
    "\n    Returns basic linker command.\n\n    Args:\n        vs:\n            Windows only. A `wdev.WindowsVS` instance or None to use default\n            `wdev.WindowsVS` instance.\n        pythonflags:\n            A `pipcl.PythonFlags` instance or None to use default\n            `pipcl.PythonFlags` instance.\n        cpp:\n            If true we return C++ linker command instead of C. On Windows this\n            has no effect - we always return `link.exe`.\n        use_env:\n            If true we use `os.environ['LD']` if set.\n\n    Returns `(linker, pythonflags)`:\n        linker:\n            Linker command. On Windows this is of the form\n            `{vs.vcvars}&&{vs.link}`; otherwise it is typically `cc` or `c++`.\n        pythonflags:\n            The `pythonflags` arg or a new `pipcl.PythonFlags` instance.\n    "
    if not pythonflags:
        pythonflags = PythonFlags()
    linker = None
    if use_env:
        if os.environ.get('LD'):
            linker = '$LD'
    if linker:
        pass
    elif windows():
        if not vs:
            vs = wdev.WindowsVS()
        linker = f'"{vs.vcvars}"&&"{vs.link}"'
    elif wasm():
        linker = 'em++' if cpp else 'emcc'
    else:
        linker = 'c++' if cpp else 'cc'
    linker = macos_add_cross_flags(linker)
    return (linker, pythonflags)

def git_items(directory, submodules=False):
    if False:
        print('Hello World!')
    '\n    Returns list of paths for all files known to git within a `directory`.\n\n    Args:\n        directory:\n            Must be somewhere within a git checkout.\n        submodules:\n            If true we also include git submodules.\n\n    Returns:\n        A list of paths for all files known to git within `directory`. Each\n        path is relative to `directory`. `directory` must be somewhere within a\n        git checkout.\n\n    We run a `git ls-files` command internally.\n\n    This function can be useful for the `fn_sdist()` callback.\n    '
    command = 'cd ' + directory + ' && git ls-files'
    if submodules:
        command += ' --recurse-submodules'
    log1(f'Running command={command!r}')
    text = subprocess.check_output(command, shell=True)
    ret = []
    for path in text.decode('utf8').strip().split('\n'):
        path2 = os.path.join(directory, path)
        if not os.path.exists(path2):
            log2(f'Ignoring git ls-files item that does not exist: {path2}')
        elif os.path.isdir(path2):
            log2(f'Ignoring git ls-files item that is actually a directory: {path2}')
        else:
            ret.append(path)
    return ret

def run(command, capture=False, check=1):
    if False:
        i = 10
        return i + 15
    '\n    Runs a command using `subprocess.run()`.\n\n    Args:\n        command:\n            A string, the command to run.\n\n            Multiple lines in `command` are are treated as a single command.\n\n            * If a line starts with `#` it is discarded.\n            * If a line contains ` #`, the trailing text is discarded.\n\n            When running the command, on Windows newlines are replaced by\n            spaces; otherwise each line is terminated by a backslash character.\n        capture:\n            If true, we return output from command.\n    Returns:\n        None on success, otherwise raises an exception.\n    '
    lines = _command_lines(command)
    nl = '\n'
    log2(f'Running: {nl.join(lines)}')
    sep = ' ' if windows() else '\\\n'
    command2 = sep.join(lines)
    if capture:
        return subprocess.run(command2, shell=True, capture_output=True, check=check, encoding='utf8').stdout
    else:
        subprocess.run(command2, shell=True, check=check)

def darwin():
    if False:
        return 10
    return sys.platform.startswith('darwin')

def windows():
    if False:
        while True:
            i = 10
    return platform.system() == 'Windows'

def wasm():
    if False:
        i = 10
        return i + 15
    return os.environ.get('OS') in ('wasm', 'wasm-mt')

def pyodide():
    if False:
        while True:
            i = 10
    return os.environ.get('PYODIDE') == '1'

def linux():
    if False:
        for i in range(10):
            print('nop')
    return platform.system() == 'Linux'

class PythonFlags:
    """
    Compile/link flags for the current python, for example the include path
    needed to get `Python.h`.

    Members:
        .includes:
            String containing compiler flags for include paths.
        .ldflags:
            String containing linker flags for library paths.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        if windows():
            wp = wdev.WindowsPython()
            self.includes = f'/I"{wp.root}\\include"'
            self.ldflags = f'/LIBPATH:"{wp.root}\\libs"'
        elif pyodide():
            _include_dir = os.environ['PYO3_CROSS_INCLUDE_DIR']
            _lib_dir = os.environ['PYO3_CROSS_LIB_DIR']
            self.includes = f'-I {_include_dir}'
            self.ldflags = f'-L {_lib_dir}'
        else:
            python_exe = os.path.realpath(sys.executable)
            if darwin():
                python_config = None
                for pc in (f'python3-config', f"{sys.executable} {sysconfig.get_config_var('srcdir')}/python-config.py", f'{python_exe}-config'):
                    e = subprocess.run(f'{pc} --includes', shell=1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=0).returncode
                    log1(f'e={e!r} from {pc!r}.')
                    if e == 0:
                        python_config = pc
                assert python_config, f'Cannot find python-config'
            else:
                python_config = f'{python_exe}-config'
            log1(f'Using python_config={python_config!r}.')
            self.includes = run(f'{python_config} --includes', capture=1).strip()
            self.ldflags = run(f'{python_config} --ldflags', capture=1).strip()
            if linux():
                ldflags2 = self.ldflags.replace(' -lcrypt ', ' ')
                if ldflags2 != self.ldflags:
                    log2(f'### Have removed `-lcrypt` from ldflags: {self.ldflags!r} -> {ldflags2!r}')
                    self.ldflags = ldflags2
        log2(f'self.includes={self.includes!r}')
        log2(f'self.ldflags={self.ldflags!r}')

def macos_add_cross_flags(command):
    if False:
        return 10
    '\n    If running on MacOS and environment variables ARCHFLAGS is set\n    (indicating we are cross-building, e.g. for arm64), returns\n    `command` with extra flags appended. Otherwise returns unchanged\n    `command`.\n    '
    if darwin():
        archflags = os.environ.get('ARCHFLAGS')
        if archflags:
            command = f'{command} {archflags}'
            log2(f'Appending ARCHFLAGS to command: {command}')
            return command
    return command

def macos_patch(library, *sublibraries):
    if False:
        while True:
            i = 10
    '\n    If running on MacOS, patches `library` so that all references to items in\n    `sublibraries` are changed to `@rpath/{leafname}`. Does nothing on other\n    platforms.\n\n    library:\n        Path of shared library.\n    sublibraries:\n        List of paths of shared libraries; these have typically been\n        specified with `-l` when `library` was created.\n    '
    log2(f'macos_patch(): library={library}  sublibraries={sublibraries}')
    if not darwin():
        return
    subprocess.run(f'otool -L {library}', shell=1, check=1)
    command = 'install_name_tool'
    names = []
    for sublibrary in sublibraries:
        name = subprocess.run(f'otool -D {sublibrary}', shell=1, check=1, capture_output=1, encoding='utf8').stdout.strip()
        name = name.split('\n')
        assert len(name) == 2 and name[0] == f'{sublibrary}:', f'name={name!r}'
        name = name[1]
        leaf = os.path.basename(name)
        m = re.match('^(.+[.]((so)|(dylib)))[0-9.]*$', leaf)
        assert m
        log2(f'Changing leaf={leaf!r} to {m.group(1)}')
        leaf = m.group(1)
        command += f' -change {name} @rpath/{leaf}'
    command += f' {library}'
    log2(f'Running: {command}')
    subprocess.run(command, shell=1, check=1)
    subprocess.run(f'otool -L {library}', shell=1, check=1)

def _command_lines(command):
    if False:
        print('Hello World!')
    '\n    Process multiline command by running through `textwrap.dedent()`, removes\n    comments (lines starting with `#` or ` #` until end of line), removes\n    entirely blank lines.\n\n    Returns list of lines.\n    '
    command = textwrap.dedent(command)
    lines = []
    for line in command.split('\n'):
        if line.startswith('#'):
            h = 0
        else:
            h = line.find(' #')
        if h >= 0:
            line = line[:h]
        if line.strip():
            lines.append(line.rstrip())
    return lines

def _cpu_name():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns `x32` or `x64` depending on Python build.\n    '
    return f'x{(32 if sys.maxsize == 2 ** 31 - 1 else 64)}'

def run_if(command, out, *prerequisites):
    if False:
        print('Hello World!')
    "\n    Runs a command only if the output file is not up to date.\n\n    Args:\n        command:\n            The command to run. We write this into a file <out>.cmd so that we\n            know to run a command if the command itself has changed.\n        out:\n            Path of the output file.\n\n        prerequisites:\n            List of prerequisite paths or true/false/None items. If an item\n            is None it is ignored, otherwise if an item is not a string we\n            immediately return it cast to a bool.\n\n    Returns:\n        True if we ran the command, otherwise None.\n\n\n    If the output file does not exist, the command is run:\n\n        >>> verbose(1)\n        1\n        >>> out = 'run_if_test_out'\n        >>> if os.path.exists( out):\n        ...     os.remove( out)\n        >>> run_if( f'touch {out}', out)\n        True\n\n    If we repeat, the output file will be up to date so the command is not run:\n\n        >>> run_if( f'touch {out}', out)\n\n    If we change the command, the command is run:\n\n        >>> run_if( f'touch  {out}', out)\n        True\n\n    If we add a prerequisite that is newer than the output, the command is run:\n\n        >>> prerequisite = 'run_if_test_prerequisite'\n        >>> run( f'touch {prerequisite}')\n        >>> run_if( f'touch {out}', out, prerequisite)\n        True\n\n    If we repeat, the output will be newer than the prerequisite, so the\n    command is not run:\n\n        >>> run_if( f'touch {out}', out, prerequisite)\n    "
    doit = False
    if not doit:
        out_mtime = _fs_mtime(out)
        if out_mtime == 0:
            doit = 'File does not exist: {out!e}'
    cmd_path = f'{out}.cmd'
    if os.path.isfile(cmd_path):
        with open(cmd_path) as f:
            cmd = f.read()
    else:
        cmd = None
    if command != cmd:
        if cmd is None:
            doit = 'No previous command stored'
        else:
            doit = f'Command has changed'
            if 0:
                doit += f': {cmd!r} => {command!r}'
    if not doit:

        def _make_prerequisites(p):
            if False:
                while True:
                    i = 10
            if isinstance(p, (list, tuple)):
                return list(p)
            else:
                return [p]
        prerequisites_all = list()
        for p in prerequisites:
            prerequisites_all += _make_prerequisites(p)
        if 0:
            log2('prerequisites_all:')
            for i in prerequisites_all:
                log2(f'    {i!r}')
        pre_mtime = 0
        pre_path = None
        for prerequisite in prerequisites_all:
            if isinstance(prerequisite, str):
                mtime = _fs_mtime_newest(prerequisite)
                if mtime >= pre_mtime:
                    pre_mtime = mtime
                    pre_path = prerequisite
            elif prerequisite is None:
                pass
            elif prerequisite:
                doit = str(prerequisite)
                break
        if not doit:
            if pre_mtime > out_mtime:
                doit = f'Prerequisite is new: {pre_path!r}'
    if doit:
        try:
            os.remove(cmd_path)
        except Exception:
            pass
        log2(f'Running command because: {doit}')
        run(command)
        with open(cmd_path, 'w') as f:
            f.write(command)
        return True
    else:
        log2(f'Not running command because up to date: {out!r}')
    if 0:
        log2(f'out_mtime={time.ctime(out_mtime)} pre_mtime={time.ctime(pre_mtime)}. pre_path={pre_path!r}: returning {ret!r}.')

def _get_prerequisites(path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns list of prerequisites from Makefile-style dependency file, e.g.\n    created by `cc -MD -MF <path>`.\n    '
    ret = list()
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                for item in line.split():
                    if item.endswith((':', '\\')):
                        continue
                    ret.append(item)
    return ret

def _fs_mtime_newest(path):
    if False:
        print('Hello World!')
    '\n    path:\n        If a file, returns mtime of the file. If a directory, returns mtime of\n        newest file anywhere within directory tree. Otherwise returns 0.\n    '
    ret = 0
    if os.path.isdir(path):
        for (dirpath, dirnames, filenames) in os.walk(path):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                ret = max(ret, _fs_mtime(path))
    else:
        ret = _fs_mtime(path)
    return ret

def _flags(items, prefix='', quote=''):
    if False:
        while True:
            i = 10
    '\n    Turns sequence into string, prefixing/quoting each item.\n    '
    if not items:
        return ''
    if isinstance(items, str):
        return items
    ret = ''
    for item in items:
        if ret:
            ret += ' '
        ret += f'{prefix}{quote}{item}{quote}'
    return ret.strip()

def _fs_mtime(filename, default=0):
    if False:
        print('Hello World!')
    "\n    Returns mtime of file, or `default` if error - e.g. doesn't exist.\n    "
    try:
        return os.path.getmtime(filename)
    except OSError:
        return default
g_verbose = int(os.environ.get('PIPCL_VERBOSE', '2'))

def verbose(level=None):
    if False:
        return 10
    '\n    Sets verbose level if `level` is not None.\n    Returns verbose level.\n    '
    global g_verbose
    if level is not None:
        g_verbose = level
    return g_verbose

def log0(text=''):
    if False:
        for i in range(10):
            print('nop')
    _log(text, 0)

def log1(text=''):
    if False:
        print('Hello World!')
    _log(text, 1)

def log2(text=''):
    if False:
        print('Hello World!')
    _log(text, 2)

def _log(text, level):
    if False:
        return 10
    '\n    Logs lines with prefix.\n    '
    if g_verbose >= level:
        caller = inspect.stack()[2].function
        for line in text.split('\n'):
            print(f'pipcl.py: {caller}(): {line}')
        sys.stdout.flush()

def _so_suffix():
    if False:
        for i in range(10):
            print('nop')
    "\n    Filename suffix for shared libraries is defined in pep-3149.  The\n    pep claims to only address posix systems, but the recommended\n    sysconfig.get_config_var('EXT_SUFFIX') also seems to give the\n    right string on Windows.\n    "
    return sysconfig.get_config_var('EXT_SUFFIX')

def install_dir(root=None):
    if False:
        i = 10
        return i + 15
    "\n    Returns install directory used by `install()`.\n\n    This will be `sysconfig.get_path('platlib')`, modified by `root` if not\n    None.\n    "
    root2 = sysconfig.get_path('platlib')
    if root:
        if windows():
            return root
        else:
            return os.path.join(root, root2.lstrip(os.sep))
    else:
        return root2

class _Record:
    """
    Internal - builds up text suitable for writing to a RECORD item, e.g.
    within a wheel.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.text = ''

    def add_content(self, content, to_):
        if False:
            return 10
        if isinstance(content, str):
            content = content.encode('utf8')
        h = hashlib.sha256(content)
        digest = h.digest()
        digest = base64.urlsafe_b64encode(digest)
        digest = digest.rstrip(b'=')
        digest = digest.decode('utf8')
        self.text += f'{to_},sha256={digest},{len(content)}\n'
        log2(f'Adding {to_}')

    def add_file(self, from_, to_):
        if False:
            print('Hello World!')
        with open(from_, 'rb') as f:
            content = f.read()
        self.add_content(content, to_)
        log2(f'Adding file: {os.path.relpath(from_)} => {to_}')

    def get(self, record_path=None):
        if False:
            return 10
        '\n        Returns contents of the RECORD file. If `record_path` is\n        specified we append a final line `<record_path>,,`; this can be\n        used to include the RECORD file itself in the contents, with\n        empty hash and size fields.\n        '
        ret = self.text
        if record_path:
            ret += f'{record_path},,\n'
        return ret