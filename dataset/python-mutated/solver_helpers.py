"""Helpers for testing the solver."""
from __future__ import annotations
import collections
import functools
import json
import pathlib
from tempfile import TemporaryDirectory
import pytest
from ..base.context import context
from ..core.solve import Solver
from ..exceptions import PackagesNotFoundError, ResolvePackageNotFound, UnsatisfiableError
from ..models.channel import Channel
from ..models.match_spec import MatchSpec
from ..models.records import PackageRecord
from . import helpers

@functools.lru_cache
def index_packages(num):
    if False:
        for i in range(10):
            print('nop')
    'Get the index data of the ``helpers.get_index_r_*`` helpers.'
    get_index = getattr(helpers, f'get_index_r_{num}')
    (index, _) = get_index(context.subdir)
    return list(index.values())

def package_string(record):
    if False:
        return 10
    return f'{record.channel.name}::{record.name}-{record.version}-{record.build}'

def package_string_set(packages):
    if False:
        while True:
            i = 10
    'Transforms package container in package string set.'
    return {package_string(record) for record in packages}

def package_dict(packages):
    if False:
        i = 10
        return i + 15
    'Transforms package container into a dictionary.'
    return {record.name: record for record in packages}

class SimpleEnvironment:
    """Helper environment object."""
    REPO_DATA_KEYS = ('build', 'build_number', 'depends', 'license', 'md5', 'name', 'sha256', 'size', 'subdir', 'timestamp', 'version', 'track_features', 'features')

    def __init__(self, path, solver_class, subdirs=context.subdirs):
        if False:
            i = 10
            return i + 15
        self._path = pathlib.Path(path)
        self._prefix_path = self._path / 'prefix'
        self._channels_path = self._path / 'channels'
        self._solver_class = solver_class
        self.subdirs = subdirs
        self.installed_packages = []
        self.repo_packages: list[str] | dict[str, list[str]] = []

    def solver(self, add, remove):
        if False:
            for i in range(10):
                print('nop')
        'Writes ``repo_packages`` to the disk and creates a solver instance.'
        channels = []
        self._write_installed_packages()
        for (channel_name, packages) in self._channel_packages.items():
            self._write_repo_packages(channel_name, packages)
            channel = Channel(str(self._channels_path / channel_name))
            channels.append(channel)
        return self._solver_class(prefix=self._prefix_path, subdirs=self.subdirs, channels=channels, specs_to_add=add, specs_to_remove=remove)

    def solver_transaction(self, add=(), remove=(), as_specs=False):
        if False:
            return 10
        packages = self.solver(add=add, remove=remove).solve_final_state()
        if as_specs:
            return packages
        return package_string_set(packages)

    def install(self, *specs, as_specs=False):
        if False:
            for i in range(10):
                print('nop')
        return self.solver_transaction(add=specs, as_specs=as_specs)

    def remove(self, *specs, as_specs=False):
        if False:
            while True:
                i = 10
        return self.solver_transaction(remove=specs, as_specs=as_specs)

    @property
    def _channel_packages(self):
        if False:
            i = 10
            return i + 15
        'Helper that unfolds the ``repo_packages`` into a dictionary.'
        if isinstance(self.repo_packages, dict):
            return self.repo_packages
        return {'test': self.repo_packages}

    def _package_data(self, record):
        if False:
            return 10
        'Turn record into data, to be written in the JSON environment/repo files.'
        data = {key: value for (key, value) in vars(record).items() if key in self.REPO_DATA_KEYS}
        if 'subdir' not in data:
            data['subdir'] = context.subdir
        return data

    def _write_installed_packages(self):
        if False:
            print('Hello World!')
        if not self.installed_packages:
            return
        conda_meta = self._prefix_path / 'conda-meta'
        conda_meta.mkdir(exist_ok=True, parents=True)
        for record in self.installed_packages:
            record_path = conda_meta / f'{record.name}-{record.version}-{record.build}.json'
            record_data = self._package_data(record)
            record_data['channel'] = record.channel.name
            record_path.write_text(json.dumps(record_data))
        history_path = conda_meta / 'history'
        history_path.write_text('\n'.join(('==> 2000-01-01 00:00:00 <==', *map(package_string, self.installed_packages))))

    def _write_repo_packages(self, channel_name, packages):
        if False:
            for i in range(10):
                print('nop')
        'Write packages to the channel path.'
        package_data = collections.defaultdict(dict)
        for record in packages:
            package_data[record.subdir][record.fn] = self._package_data(record)
        assert set(self.subdirs).issuperset(set(package_data.keys()))
        for subdir in self.subdirs:
            subdir_path = self._channels_path / channel_name / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)
            subdir_path.joinpath('repodata.json').write_text(json.dumps({'info': {'subdir': subdir}, 'packages': package_data.get(subdir, {})}))

def empty_prefix():
    if False:
        while True:
            i = 10
    return TemporaryDirectory(prefix='conda-test-repo-')

@pytest.fixture()
def temp_simple_env(solver_class=Solver) -> SimpleEnvironment:
    if False:
        return 10
    with empty_prefix() as prefix:
        yield SimpleEnvironment(prefix, solver_class)

class SolverTests:
    """Tests for :py:class:`conda.core.solve.Solver` implementations."""

    @property
    def solver_class(self) -> type[Solver]:
        if False:
            i = 10
            return i + 15
        'Class under test.'
        raise NotImplementedError

    @property
    def tests_to_skip(self):
        if False:
            while True:
                i = 10
        return {}

    @pytest.fixture(autouse=True)
    def skip_tests(self, request):
        if False:
            while True:
                i = 10
        for (reason, skip_list) in self.tests_to_skip.items():
            if request.node.name in skip_list:
                pytest.skip(reason)

    @pytest.fixture()
    def env(self):
        if False:
            for i in range(10):
                print('nop')
        with TemporaryDirectory(prefix='conda-test-repo-') as tmpdir:
            self.env = SimpleEnvironment(tmpdir, self.solver_class)
            yield self.env
            self.env = None

    def find_package_in_list(self, packages, **kwargs):
        if False:
            return 10
        for record in packages:
            if all((getattr(record, key) == value for (key, value) in kwargs.items())):
                return record

    def find_package(self, **kwargs):
        if False:
            while True:
                i = 10
        if isinstance(self.env.repo_packages, dict):
            if 'channel' not in kwargs:
                raise ValueError('Repo has multiple channels, the `channel` argument must be specified')
            packages = self.env.repo_packages[kwargs['channel']]
        else:
            packages = self.env.repo_packages
        return self.find_package_in_list(packages, **kwargs)

    def assert_unsatisfiable(self, exc_info, entries):
        if False:
            for i in range(10):
                print('nop')
        'Helper to assert that a :py:class:`conda.exceptions.UnsatisfiableError`\n        instance as a the specified set of unsatisfiable specifications.\n        '
        assert issubclass(exc_info.type, UnsatisfiableError)
        if exc_info.type is UnsatisfiableError:
            assert sorted((tuple(map(str, entries)) for entries in exc_info.value.unsatisfiable)) == entries

    def test_empty(self, env):
        if False:
            i = 10
            return i + 15
        env.repo_packages = index_packages(1)
        assert env.install() == set()

    def test_iopro_mkl(self, env):
        if False:
            for i in range(10):
                print('nop')
        env.repo_packages = index_packages(1)
        assert env.install('iopro 1.4*', 'python 2.7*', 'numpy 1.7*') == {'test::iopro-1.4.3-np17py27_p0', 'test::numpy-1.7.1-py27_0', 'test::openssl-1.0.1c-0', 'test::python-2.7.5-0', 'test::readline-6.2-0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::unixodbc-2.3.1-0', 'test::zlib-1.2.7-0', 'test::distribute-0.6.36-py27_1', 'test::pip-1.3.1-py27_1'}

    def test_iopro_nomkl(self, env):
        if False:
            print('Hello World!')
        env.repo_packages = index_packages(1)
        assert env.install('iopro 1.4*', 'python 2.7*', 'numpy 1.7*', MatchSpec(track_features='mkl')) == {'test::iopro-1.4.3-np17py27_p0', 'test::mkl-rt-11.0-p0', 'test::numpy-1.7.1-py27_p0', 'test::openssl-1.0.1c-0', 'test::python-2.7.5-0', 'test::readline-6.2-0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::unixodbc-2.3.1-0', 'test::zlib-1.2.7-0', 'test::distribute-0.6.36-py27_1', 'test::pip-1.3.1-py27_1'}

    def test_mkl(self, env):
        if False:
            for i in range(10):
                print('nop')
        env.repo_packages = index_packages(1)
        assert env.install('mkl') == env.install('mkl 11*', MatchSpec(track_features='mkl'))

    def test_accelerate(self, env):
        if False:
            i = 10
            return i + 15
        env.repo_packages = index_packages(1)
        assert env.install('accelerate') == env.install('accelerate', MatchSpec(track_features='mkl'))

    def test_scipy_mkl(self, env):
        if False:
            print('Hello World!')
        env.repo_packages = index_packages(1)
        records = env.install('scipy', 'python 2.7*', 'numpy 1.7*', MatchSpec(track_features='mkl'), as_specs=True)
        for record in records:
            if record.name in ('numpy', 'scipy'):
                assert 'mkl' in record.features
        assert 'test::numpy-1.7.1-py27_p0' in package_string_set(records)
        assert 'test::scipy-0.12.0-np17py27_p0' in package_string_set(records)

    def test_anaconda_nomkl(self, env):
        if False:
            for i in range(10):
                print('nop')
        env.repo_packages = index_packages(1)
        records = env.install('anaconda 1.5.0', 'python 2.7*', 'numpy 1.7*')
        assert len(records) == 107
        assert 'test::scipy-0.12.0-np17py27_0' in records

    def test_pseudo_boolean(self, env):
        if False:
            i = 10
            return i + 15
        env.repo_packages = index_packages(1)
        assert env.install('iopro', 'python 2.7*', 'numpy 1.5*') == {'test::iopro-1.4.3-np15py27_p0', 'test::numpy-1.5.1-py27_4', 'test::openssl-1.0.1c-0', 'test::python-2.7.5-0', 'test::readline-6.2-0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::unixodbc-2.3.1-0', 'test::zlib-1.2.7-0', 'test::distribute-0.6.36-py27_1', 'test::pip-1.3.1-py27_1'}
        assert env.install('iopro', 'python 2.7*', 'numpy 1.5*', MatchSpec(track_features='mkl')) == {'test::iopro-1.4.3-np15py27_p0', 'test::mkl-rt-11.0-p0', 'test::numpy-1.5.1-py27_p4', 'test::openssl-1.0.1c-0', 'test::python-2.7.5-0', 'test::readline-6.2-0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::unixodbc-2.3.1-0', 'test::zlib-1.2.7-0', 'test::distribute-0.6.36-py27_1', 'test::pip-1.3.1-py27_1'}

    def test_unsat_from_r1(self, env):
        if False:
            i = 10
            return i + 15
        env.repo_packages = index_packages(1)
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('numpy 1.5*', 'scipy 0.12.0b1')
        self.assert_unsatisfiable(exc_info, [('numpy=1.5',), ('scipy==0.12.0b1', "numpy[version='1.6.*|1.7.*']")])
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('numpy 1.5*', 'python 3*')
        self.assert_unsatisfiable(exc_info, [('numpy=1.5', 'nose', 'python=3.3'), ('numpy=1.5', "python[version='2.6.*|2.7.*']"), ('python=3',)])
        with pytest.raises((ResolvePackageNotFound, PackagesNotFoundError)) as exc_info:
            env.install('numpy 1.5*', 'numpy 1.6*')
        if exc_info.type is ResolvePackageNotFound:
            assert sorted(map(str, exc_info.value.bad_deps)) == ["numpy[version='1.5.*,1.6.*']"]

    def test_unsat_simple(self, env):
        if False:
            while True:
                i = 10
        env.repo_packages = [helpers.record(name='a', depends=['c >=1,<2']), helpers.record(name='b', depends=['c >=2,<3']), helpers.record(name='c', version='1.0'), helpers.record(name='c', version='2.0')]
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('a', 'b')
        self.assert_unsatisfiable(exc_info, [('a', "c[version='>=1,<2']"), ('b', "c[version='>=2,<3']")])

    def test_get_dists(self, env):
        if False:
            print('Hello World!')
        env.repo_packages = index_packages(1)
        records = env.install('anaconda 1.4.0')
        assert 'test::anaconda-1.4.0-np17py33_0' in records
        assert 'test::freetype-2.4.10-0' in records

    def test_unsat_shortest_chain_1(self, env):
        if False:
            i = 10
            return i + 15
        env.repo_packages = [helpers.record(name='a', depends=['d', 'c <1.3.0']), helpers.record(name='b', depends=['c']), helpers.record(name='c', version='1.3.6'), helpers.record(name='c', version='1.2.8'), helpers.record(name='d', depends=['c >=0.8.0'])]
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('c=1.3.6', 'a', 'b')
        self.assert_unsatisfiable(exc_info, [('a', "c[version='<1.3.0']"), ('a', 'd', "c[version='>=0.8.0']"), ('b', 'c'), ('c=1.3.6',)])

    def test_unsat_shortest_chain_2(self, env):
        if False:
            while True:
                i = 10
        env.repo_packages = [helpers.record(name='a', depends=['d', 'c >=0.8.0']), helpers.record(name='b', depends=['c']), helpers.record(name='c', version='1.3.6'), helpers.record(name='c', version='1.2.8'), helpers.record(name='d', depends=['c <1.3.0'])]
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('c=1.3.6', 'a', 'b')
        self.assert_unsatisfiable(exc_info, [('a', "c[version='>=0.8.0']"), ('a', 'd', "c[version='<1.3.0']"), ('b', 'c'), ('c=1.3.6',)])

    def test_unsat_shortest_chain_3(self, env):
        if False:
            i = 10
            return i + 15
        env.repo_packages = [helpers.record(name='a', depends=['f', 'e']), helpers.record(name='b', depends=['c']), helpers.record(name='c', version='1.3.6'), helpers.record(name='c', version='1.2.8'), helpers.record(name='d', depends=['c >=0.8.0']), helpers.record(name='e', depends=['c <1.3.0']), helpers.record(name='f', depends=['d'])]
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('c=1.3.6', 'a', 'b')
        self.assert_unsatisfiable(exc_info, [('a', 'e', "c[version='<1.3.0']"), ('b', 'c'), ('c=1.3.6',)])

    def test_unsat_shortest_chain_4(self, env):
        if False:
            while True:
                i = 10
        env.repo_packages = [helpers.record(name='a', depends=['py =3.7.1']), helpers.record(name='py_req_1'), helpers.record(name='py_req_2'), helpers.record(name='py', version='3.7.1', depends=['py_req_1', 'py_req_2']), helpers.record(name='py', version='3.6.1', depends=['py_req_1', 'py_req_2'])]
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('a', 'py=3.6.1')
        self.assert_unsatisfiable(exc_info, [('a', 'py=3.7.1'), ('py=3.6.1',)])

    def test_unsat_chain(self, env):
        if False:
            print('Hello World!')
        env.repo_packages = [helpers.record(name='a', depends=['b']), helpers.record(name='b', depends=['c >=1,<2']), helpers.record(name='c', version='1.0', depends=['d >=1,<2']), helpers.record(name='d', version='1.0'), helpers.record(name='e', depends=['c >=2,<3']), helpers.record(name='c', version='2.0', depends=['d >=2,<3']), helpers.record(name='d', version='2.0')]
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('a', 'e')
        self.assert_unsatisfiable(exc_info, [('a', 'b', "c[version='>=1,<2']"), ('e', "c[version='>=2,<3']")])

    def test_unsat_any_two_not_three(self, env):
        if False:
            print('Hello World!')
        env.repo_packages = [helpers.record(name='a', version='1.0', depends=['d >=1,<2']), helpers.record(name='a', version='2.0', depends=['d >=2,<3']), helpers.record(name='b', version='1.0', depends=['d >=1,<2']), helpers.record(name='b', version='2.0', depends=['d >=3,<4']), helpers.record(name='c', version='1.0', depends=['d >=2,<3']), helpers.record(name='c', version='2.0', depends=['d >=3,<4']), helpers.record(name='d', version='1.0'), helpers.record(name='d', version='2.0'), helpers.record(name='d', version='3.0')]
        installed = env.install('a', 'b', as_specs=True)
        assert any((k.name == 'a' and k.version == '1.0' for k in installed))
        assert any((k.name == 'b' and k.version == '1.0' for k in installed))
        installed = env.install('a', 'c', as_specs=True)
        assert any((k.name == 'a' and k.version == '2.0' for k in installed))
        assert any((k.name == 'c' and k.version == '1.0' for k in installed))
        installed = env.install('b', 'c', as_specs=True)
        assert any((k.name == 'b' and k.version == '2.0' for k in installed))
        assert any((k.name == 'c' and k.version == '2.0' for k in installed))
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('a', 'b', 'c')
        self.assert_unsatisfiable(exc_info, [('a', "d[version='>=1,<2|>=2,<3']"), ('b', "d[version='>=1,<2|>=3,<4']"), ('c', "d[version='>=2,<3|>=3,<4']")])

    def test_unsat_expand_single(self, env):
        if False:
            while True:
                i = 10
        env.repo_packages = [helpers.record(name='a', depends=['b', 'c']), helpers.record(name='b', depends=['d >=1,<2']), helpers.record(name='c', depends=['d >=2,<3']), helpers.record(name='d', version='1.0'), helpers.record(name='d', version='2.0')]
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('a')
        self.assert_unsatisfiable(exc_info, [('b', "d[version='>=1,<2']"), ('c', "d[version='>=2,<3']")])

    def test_unsat_missing_dep(self, env):
        if False:
            return 10
        env.repo_packages = [helpers.record(name='a', depends=['b', 'c']), helpers.record(name='b', depends=['c >=2,<3']), helpers.record(name='c', version='1.0')]
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('a', 'b')
        self.assert_unsatisfiable(exc_info, [('a', 'b'), ('b',)])

    def test_nonexistent(self, env):
        if False:
            while True:
                i = 10
        with pytest.raises((ResolvePackageNotFound, PackagesNotFoundError)):
            env.install('notarealpackage 2.0*')
        with pytest.raises((ResolvePackageNotFound, PackagesNotFoundError)):
            env.install('numpy 1.5')

    def test_timestamps_and_deps(self, env):
        if False:
            print('Hello World!')
        env.repo_packages = index_packages(1) + [helpers.record(name='mypackage', version='1.0', build='hash12_0', timestamp=1, depends=['libpng 1.2.*']), helpers.record(name='mypackage', version='1.0', build='hash15_0', timestamp=0, depends=['libpng 1.5.*'])]
        records_12 = env.install('libpng 1.2.*', 'mypackage')
        assert 'test::libpng-1.2.50-0' in records_12
        assert 'test::mypackage-1.0-hash12_0' in records_12
        records_15 = env.install('libpng 1.5.*', 'mypackage')
        assert 'test::libpng-1.5.13-1' in records_15
        assert 'test::mypackage-1.0-hash15_0' in records_15
        assert env.install('mypackage', *env.install('libpng 1.2.*', as_specs=True)) == records_12
        assert env.install('mypackage', *env.install('libpng 1.5.*', as_specs=True)) == records_15
        assert env.install('mypackage') == records_15

    def test_nonexistent_deps(self, env):
        if False:
            i = 10
            return i + 15
        env.repo_packages = index_packages(1) + [helpers.record(name='mypackage', version='1.0', depends=['nose', 'python 3.3*', 'notarealpackage 2.0*']), helpers.record(name='mypackage', version='1.1', depends=['nose', 'python 3.3*']), helpers.record(name='anotherpackage', version='1.0', depends=['nose', 'mypackage 1.1']), helpers.record(name='anotherpackage', version='2.0', depends=['nose', 'mypackage'])]
        assert env.install('mypackage') == {'test::mypackage-1.1-0', 'test::nose-1.3.0-py33_0', 'test::openssl-1.0.1c-0', 'test::python-3.3.2-0', 'test::readline-6.2-0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0', 'test::distribute-0.6.36-py33_1', 'test::pip-1.3.1-py33_1'}
        assert env.install('anotherpackage 1.0') == {'test::anotherpackage-1.0-0', 'test::mypackage-1.1-0', 'test::nose-1.3.0-py33_0', 'test::openssl-1.0.1c-0', 'test::python-3.3.2-0', 'test::readline-6.2-0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0', 'test::distribute-0.6.36-py33_1', 'test::pip-1.3.1-py33_1'}
        assert env.install('anotherpackage') == {'test::anotherpackage-2.0-0', 'test::mypackage-1.1-0', 'test::nose-1.3.0-py33_0', 'test::openssl-1.0.1c-0', 'test::python-3.3.2-0', 'test::readline-6.2-0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0', 'test::distribute-0.6.36-py33_1', 'test::pip-1.3.1-py33_1'}
        env.repo_packages = index_packages(1) + [helpers.record(name='mypackage', version='1.0', depends=['nose', 'python 3.3*']), helpers.record(name='mypackage', version='1.1', depends=['nose', 'python 3.3*', 'notarealpackage 2.0*']), helpers.record(name='anotherpackage', version='1.0', depends=['nose', 'mypackage 1.0']), helpers.record(name='anotherpackage', version='2.0', depends=['nose', 'mypackage'])]
        assert env.install('mypackage') == {'test::mypackage-1.0-0', 'test::nose-1.3.0-py33_0', 'test::openssl-1.0.1c-0', 'test::python-3.3.2-0', 'test::readline-6.2-0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0', 'test::distribute-0.6.36-py33_1', 'test::pip-1.3.1-py33_1'}
        with pytest.raises((ResolvePackageNotFound, UnsatisfiableError)):
            env.install('mypackage 1.1')
        assert env.install('anotherpackage 1.0') == {'test::anotherpackage-1.0-0', 'test::mypackage-1.0-0', 'test::nose-1.3.0-py33_0', 'test::openssl-1.0.1c-0', 'test::python-3.3.2-0', 'test::readline-6.2-0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0', 'test::distribute-0.6.36-py33_1', 'test::pip-1.3.1-py33_1'}
        assert env.install('anotherpackage') == {'test::anotherpackage-2.0-0', 'test::mypackage-1.0-0', 'test::nose-1.3.0-py33_0', 'test::openssl-1.0.1c-0', 'test::python-3.3.2-0', 'test::readline-6.2-0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0', 'test::distribute-0.6.36-py33_1', 'test::pip-1.3.1-py33_1'}

    def test_install_package_with_feature(self, env):
        if False:
            while True:
                i = 10
        env.repo_packages = index_packages(1) + [helpers.record(name='mypackage', version='1.0', depends=['python 3.3*'], features='feature'), helpers.record(name='feature', version='1.0', depends=['python 3.3*'], track_features='feature')]
        env.install('mypackage', 'feature 1.0')

    def test_unintentional_feature_downgrade(self, env):
        if False:
            while True:
                i = 10
        good_rec_match = MatchSpec('channel-1::scipy==0.11.0=np17py33_3')
        good_rec = next((prec for prec in index_packages(1) if good_rec_match.match(prec)))
        bad_deps = tuple((d for d in good_rec.depends if not d.startswith('numpy')))
        bad_rec = PackageRecord.from_objects(good_rec, channel='test', build=good_rec.build.replace('_3', '_x0'), build_number=0, depends=bad_deps, fn=good_rec.fn.replace('_3', '_x0'), url=good_rec.url.replace('_3', '_x0'))
        env.repo_packages = index_packages(1) + [bad_rec]
        records = env.install('scipy 0.11.0')
        assert 'test::scipy-0.11.0-np17py33_x0' not in records
        assert 'test::scipy-0.11.0-np17py33_3' in records

    def test_circular_dependencies(self, env):
        if False:
            i = 10
            return i + 15
        env.repo_packages = index_packages(1) + [helpers.record(name='package1', depends=['package2']), helpers.record(name='package2', depends=['package1'])]
        assert env.install('package1', 'package2') == env.install('package1') == env.install('package2')

    def test_irrational_version(self, env):
        if False:
            while True:
                i = 10
        env.repo_packages = index_packages(1)
        assert env.install('pytz 2012d', 'python 3*') == {'test::distribute-0.6.36-py33_1', 'test::openssl-1.0.1c-0', 'test::pip-1.3.1-py33_1', 'test::python-3.3.2-0', 'test::pytz-2012d-py33_0', 'test::readline-6.2-0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0'}

    def test_no_features(self, env):
        if False:
            i = 10
            return i + 15
        env.repo_packages = index_packages(1)
        assert env.install('python 2.6*', 'numpy 1.6*', 'scipy 0.11*') == {'test::distribute-0.6.36-py26_1', 'test::numpy-1.6.2-py26_4', 'test::openssl-1.0.1c-0', 'test::pip-1.3.1-py26_1', 'test::python-2.6.8-6', 'test::readline-6.2-0', 'test::scipy-0.11.0-np16py26_3', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0'}
        assert env.install('python 2.6*', 'numpy 1.6*', 'scipy 0.11*', MatchSpec(track_features='mkl')) == {'test::distribute-0.6.36-py26_1', 'test::mkl-rt-11.0-p0', 'test::numpy-1.6.2-py26_p4', 'test::openssl-1.0.1c-0', 'test::pip-1.3.1-py26_1', 'test::python-2.6.8-6', 'test::readline-6.2-0', 'test::scipy-0.11.0-np16py26_p3', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0'}
        env.repo_packages += [helpers.record(name='pandas', version='0.12.0', build='np16py27_0', depends=['dateutil', 'numpy 1.6*', 'python 2.7*', 'pytz']), helpers.record(name='numpy', version='1.6.2', build='py27_p5', build_number=0, depends=['mkl-rt 11.0', 'python 2.7'], features='mkl')]
        assert env.install('pandas 0.12.0 np16py27_0', 'python 2.7*') == {'test::dateutil-2.1-py27_1', 'test::distribute-0.6.36-py27_1', 'test::numpy-1.6.2-py27_4', 'test::openssl-1.0.1c-0', 'test::pandas-0.12.0-np16py27_0', 'test::pip-1.3.1-py27_1', 'test::python-2.7.5-0', 'test::pytz-2013b-py27_0', 'test::readline-6.2-0', 'test::six-1.3.0-py27_0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0'}
        assert env.install('pandas 0.12.0 np16py27_0', 'python 2.7*', MatchSpec(track_features='mkl')) == {'test::dateutil-2.1-py27_1', 'test::distribute-0.6.36-py27_1', 'test::mkl-rt-11.0-p0', 'test::numpy-1.6.2-py27_p4', 'test::openssl-1.0.1c-0', 'test::pandas-0.12.0-np16py27_0', 'test::pip-1.3.1-py27_1', 'test::python-2.7.5-0', 'test::pytz-2013b-py27_0', 'test::readline-6.2-0', 'test::six-1.3.0-py27_0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0'}

    @pytest.mark.xfail(reason='CONDA_CHANNEL_PRIORITY does not seem to have any effect')
    def test_channel_priority_1(self, monkeypatch, env):
        if False:
            return 10
        env.repo_packages = {}
        env.repo_packages['channel-A'] = []
        env.repo_packages['channel-1'] = index_packages(1)
        pandas_0 = self.find_package(channel='channel-1', name='pandas', version='0.10.1', build='np17py27_0')
        env.repo_packages['channel-A'].append(pandas_0)
        monkeypatch.setenv('CONDA_CHANNEL_PRIORITY', 'True')
        assert 'channel-A::pandas-0.11.0-np16py27_0' in env.install('pandas', 'python 2.7*', 'numpy 1.6*')
        monkeypatch.setenv('CONDA_CHANNEL_PRIORITY', 'False')
        assert 'channel-1::pandas-0.11.0-np16py27_1' in env.install('pandas', 'python 2.7*', 'numpy 1.6*')
        env.repo_packages = dict(reversed(env.repo_packages.items()))
        monkeypatch.setenv('CONDA_CHANNEL_PRIORITY', 'True')
        assert 'channel-1::pandas-0.11.0-np16py27_1' in env.install('pandas', 'python 2.7*', 'numpy 1.6*')

    @pytest.mark.xfail(reason='CONDA_CHANNEL_PRIORITY does not seem to have any effect')
    def test_unsat_channel_priority(self, monkeypatch, env):
        if False:
            for i in range(10):
                print('nop')
        env.repo_packages = {}
        env.repo_packages['channel-1'] = [helpers.record(name='a', version='1.0', depends=['c']), helpers.record(name='b', version='1.0', depends=['c >=2,<3']), helpers.record(name='c', version='1.0')]
        env.repo_packages['channel-2'] = [helpers.record(name='a', version='2.0', depends=['c']), helpers.record(name='b', version='2.0', depends=['c >=2,<3']), helpers.record(name='c', version='1.0'), helpers.record(name='c', version='2.0')]
        monkeypatch.setenv('CONDA_CHANNEL_PRIORITY', 'True')
        records = env.install('a', 'b', as_specs=True)
        assert any((k.name == 'a' and k.version == '1.0' for k in records))
        assert any((k.name == 'b' and k.version == '1.0' for k in records))
        monkeypatch.setenv('CONDA_CHANNEL_PRIORITY', 'False')
        records = env.install('a', 'b', as_specs=True)
        assert any((k.name == 'a' and k.version == '2.0' for k in records))
        assert any((k.name == 'b' and k.version == '2.0' for k in records))
        monkeypatch.setenv('CONDA_CHANNEL_PRIORITY', 'True')
        with pytest.raises(UnsatisfiableError) as exc_info:
            env.install('a', 'b')
        self.assert_unsatisfiable(exc_info, [('b', "c[version='>=2,<3']")])

    @pytest.mark.xfail(reason='There is some weird global state making this test fail when the whole test suite is run')
    def test_remove(self, env):
        if False:
            return 10
        env.repo_packages = index_packages(1)
        records = env.install('pandas', 'python 2.7*', as_specs=True)
        assert package_string_set(records) == {'test::dateutil-2.1-py27_1', 'test::distribute-0.6.36-py27_1', 'test::numpy-1.7.1-py27_0', 'test::openssl-1.0.1c-0', 'test::pandas-0.11.0-np17py27_1', 'test::pip-1.3.1-py27_1', 'test::python-2.7.5-0', 'test::pytz-2013b-py27_0', 'test::readline-6.2-0', 'test::scipy-0.12.0-np17py27_0', 'test::six-1.3.0-py27_0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0'}
        env.installed_packages = records
        assert env.remove('pandas') == {'test::dateutil-2.1-py27_1', 'test::distribute-0.6.36-py27_1', 'test::numpy-1.7.1-py27_0', 'test::openssl-1.0.1c-0', 'test::pip-1.3.1-py27_1', 'test::python-2.7.5-0', 'test::pytz-2013b-py27_0', 'test::readline-6.2-0', 'test::scipy-0.12.0-np17py27_0', 'test::six-1.3.0-py27_0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0'}
        assert env.remove('numpy') == {'test::dateutil-2.1-py27_1', 'test::distribute-0.6.36-py27_1', 'test::openssl-1.0.1c-0', 'test::pip-1.3.1-py27_1', 'test::python-2.7.5-0', 'test::pytz-2013b-py27_0', 'test::readline-6.2-0', 'test::six-1.3.0-py27_0', 'test::sqlite-3.7.13-0', 'test::system-5.8-1', 'test::tk-8.5.13-0', 'test::zlib-1.2.7-0'}

    def test_surplus_features_1(self, env):
        if False:
            i = 10
            return i + 15
        env.repo_packages += [helpers.record(name='feature', track_features='feature'), helpers.record(name='package1', features='feature'), helpers.record(name='package2', version='1.0', features='feature', depends=['package1']), helpers.record(name='package2', version='2.0', features='feature')]
        assert env.install('package2', 'feature') == {'test::package2-2.0-0', 'test::feature-1.0-0'}

    def test_surplus_features_2(self, env):
        if False:
            while True:
                i = 10
        env.repo_packages += [helpers.record(name='feature', track_features='feature'), helpers.record(name='package1', features='feature'), helpers.record(name='package2', version='1.0', build_number=0, features='feature', depends=['package1']), helpers.record(name='package2', version='1.0', build_number=1, features='feature')]
        assert env.install('package2', 'feature') == {'test::package2-1.0-0', 'test::feature-1.0-0'}

    def test_get_reduced_index_broadening_with_unsatisfiable_early_dep(self, env):
        if False:
            for i in range(10):
                print('nop')
        env.repo_packages += [helpers.record(name='a', version='1.0', depends=['b', 'c==2']), helpers.record(name='a', version='2.0', depends=['b']), helpers.record(name='b', depends=['c']), helpers.record(name='c')]
        assert env.install('a') == {'test::a-2.0-0', 'test::b-1.0-0', 'test::c-1.0-0'}

    def test_get_reduced_index_broadening_preferred_solution(self, env):
        if False:
            for i in range(10):
                print('nop')
        env.repo_packages += [helpers.record(name='top', version='1.0', depends=['middle', 'bottom==1.5']), helpers.record(name='top', version='2.0', depends=['middle']), helpers.record(name='middle', depends=['bottom']), helpers.record(name='bottom', version='1.5'), helpers.record(name='bottom', version='2.5')]
        for record in env.install('top', as_specs=True):
            if record.name == 'top':
                assert record.version == '2.0', f'top version should be 2.0, but is {record.version}'
            elif record.name == 'bottom':
                assert record.version == '2.5', f'bottom version should be 2.5, but is {record.version}'

    def test_arch_preferred_over_noarch_when_otherwise_equal(self, env):
        if False:
            for i in range(10):
                print('nop')
        env.repo_packages += [helpers.record(name='package1', subdir='noarch'), helpers.record(name='package1')]
        records = env.install('package1', as_specs=True)
        assert len(records) == 1
        assert records[0].subdir == context.subdir

    def test_noarch_preferred_over_arch_when_version_greater(self, env):
        if False:
            print('Hello World!')
        env.repo_packages += [helpers.record(name='package1', version='2.0', subdir='noarch'), helpers.record(name='package1', version='1.0')]
        records = env.install('package1', as_specs=True)
        assert len(records) == 1
        assert records[0].subdir == 'noarch'

    def test_noarch_preferred_over_arch_when_version_greater_dep(self, env):
        if False:
            print('Hello World!')
        env.repo_packages += [helpers.record(name='package1', version='1.0'), helpers.record(name='package1', version='2.0', subdir='noarch'), helpers.record(name='package2', depends=['package1'])]
        records = env.install('package2', as_specs=True)
        package1 = self.find_package_in_list(records, name='package1')
        assert package1.subdir == 'noarch'

    def test_noarch_preferred_over_arch_when_build_greater(self, env):
        if False:
            return 10
        env.repo_packages += [helpers.record(name='package1', build_number=0), helpers.record(name='package1', build_number=1, subdir='noarch')]
        records = env.install('package1', as_specs=True)
        assert len(records) == 1
        assert records[0].subdir == 'noarch'

    def test_noarch_preferred_over_arch_when_build_greater_dep(self, env):
        if False:
            i = 10
            return i + 15
        env.repo_packages += [helpers.record(name='package1', build_number=0), helpers.record(name='package1', build_number=1, subdir='noarch'), helpers.record(name='package2', depends=['package1'])]
        records = env.install('package2', as_specs=True)
        package1 = self.find_package_in_list(records, name='package1')
        assert package1.subdir == 'noarch'