import json
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple
from zipfile import ZipFile
from pip._internal.utils.urls import path_to_url
from tests.lib import PipTestEnvironment, create_basic_sdist_for_package
PYPROJECT_TOML = '[build-system]\nrequires = []\nbuild-backend = "dummy_backend:main"\nbackend-path = ["backend"]\n'
BACKEND_SRC = '\nimport csv\nimport json\nimport os.path\nfrom zipfile import ZipFile\nimport hashlib\nimport base64\nimport io\n\nWHEEL = """Wheel-Version: 1.0\nGenerator: dummy_backend 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"""\n\nMETADATA = """Metadata-Version: 2.1\nName: {project}\nVersion: {version}\nSummary: A dummy package\nAuthor: None\nAuthor-email: none@example.org\nLicense: MIT\n{requires_dist}\n"""\n\ndef make_wheel(z, project, version, requires_dist, files):\n    record = []\n    def add_file(name, data):\n        data = data.encode("utf-8")\n        z.writestr(name, data)\n        digest = hashlib.sha256(data).digest()\n        hash = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ASCII")\n        record.append((name, f"sha256={hash}", len(data)))\n    distinfo = f"{project}-{version}.dist-info"\n    add_file(f"{distinfo}/WHEEL", WHEEL)\n    add_file(f"{distinfo}/METADATA", METADATA.format(\n        project=project, version=version, requires_dist=requires_dist\n    ))\n    for name, data in files:\n        add_file(name, data)\n    record_name = f"{distinfo}/RECORD"\n    record.append((record_name, "", ""))\n    b = io.BytesIO()\n    rec = io.TextIOWrapper(b, newline="", encoding="utf-8")\n    w = csv.writer(rec)\n    w.writerows(record)\n    z.writestr(record_name, b.getvalue())\n    rec.close()\n\n\nclass Backend:\n    def build_wheel(\n        self,\n        wheel_directory,\n        config_settings=None,\n        metadata_directory=None\n    ):\n        if config_settings is None:\n            config_settings = {}\n        w = os.path.join(wheel_directory, "{{name}}-1.0-py3-none-any.whl")\n        with open(w, "wb") as f:\n            with ZipFile(f, "w") as z:\n                make_wheel(\n                    z, "{{name}}", "1.0", "{{requires_dist}}",\n                    [("{{name}}-config.json", json.dumps(config_settings))]\n                )\n        return "{{name}}-1.0-py3-none-any.whl"\n\n    build_editable = build_wheel\n\nmain = Backend()\n'

def make_project(path: Path, name: str='foo', dependencies: Optional[List[str]]=None) -> Tuple[str, str, Path]:
    if False:
        i = 10
        return i + 15
    version = '1.0'
    project_dir = path / name
    backend = project_dir / 'backend'
    backend.mkdir(parents=True)
    (project_dir / 'pyproject.toml').write_text(PYPROJECT_TOML)
    requires_dist = [f'Requires-Dist: {dep}' for dep in dependencies or []]
    (backend / 'dummy_backend.py').write_text(BACKEND_SRC.replace('{{name}}', name).replace('{{requires_dist}}', '\n'.join(requires_dist)))
    return (name, version, project_dir)

def test_backend_sees_config(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    (name, version, project_dir) = make_project(script.scratch_path)
    script.pip('wheel', '--config-settings', 'FOO=Hello', project_dir)
    wheel_file_name = f'{name}-{version}-py3-none-any.whl'
    wheel_file_path = script.cwd / wheel_file_name
    with open(wheel_file_path, 'rb') as f:
        with ZipFile(f) as z:
            output = z.read(f'{name}-config.json')
            assert json.loads(output) == {'FOO': 'Hello'}

def test_backend_sees_config_reqs(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    (name, version, project_dir) = make_project(script.scratch_path)
    script.scratch_path.joinpath('reqs.txt').write_text(f'{project_dir} --config-settings FOO=Hello')
    script.pip('wheel', '-r', 'reqs.txt')
    wheel_file_name = f'{name}-{version}-py3-none-any.whl'
    wheel_file_path = script.cwd / wheel_file_name
    with open(wheel_file_path, 'rb') as f:
        with ZipFile(f) as z:
            output = z.read(f'{name}-config.json')
            assert json.loads(output) == {'FOO': 'Hello'}

def test_backend_sees_config_via_constraint(script: PipTestEnvironment) -> None:
    if False:
        return 10
    (name, version, project_dir) = make_project(script.scratch_path)
    constraints_file = script.scratch_path / 'constraints.txt'
    constraints_file.write_text(f'{name} @ {path_to_url(str(project_dir))}')
    script.pip('wheel', '--config-settings', 'FOO=Hello', '-c', 'constraints.txt', name)
    wheel_file_name = f'{name}-{version}-py3-none-any.whl'
    wheel_file_path = script.cwd / wheel_file_name
    with open(wheel_file_path, 'rb') as f:
        with ZipFile(f) as z:
            output = z.read(f'{name}-config.json')
            assert json.loads(output) == {'FOO': 'Hello'}

def test_backend_sees_config_via_sdist(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    (name, version, project_dir) = make_project(script.scratch_path)
    dists_dir = script.scratch_path / 'dists'
    dists_dir.mkdir()
    with tarfile.open(dists_dir / f'{name}-{version}.tar.gz', 'w:gz') as dist_tar:
        dist_tar.add(project_dir, arcname=name)
    script.pip('wheel', '--config-settings', 'FOO=Hello', '-f', dists_dir, name)
    wheel_file_name = f'{name}-{version}-py3-none-any.whl'
    wheel_file_path = script.cwd / wheel_file_name
    with open(wheel_file_path, 'rb') as f:
        with ZipFile(f) as z:
            output = z.read(f'{name}-config.json')
            assert json.loads(output) == {'FOO': 'Hello'}

def test_req_file_does_not_see_config(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Test that CLI config settings do not propagate to requirement files.'
    (name, _, project_dir) = make_project(script.scratch_path)
    reqs_file = script.scratch_path / 'reqs.txt'
    reqs_file.write_text(f'{project_dir}')
    script.pip('install', '--config-settings', 'FOO=Hello', '-r', reqs_file)
    config = script.site_packages_path / f'{name}-config.json'
    with open(config, 'rb') as f:
        assert json.load(f) == {}

def test_dep_does_not_see_config(script: PipTestEnvironment) -> None:
    if False:
        i = 10
        return i + 15
    'Test that CLI config settings do not propagate to dependencies.'
    (_, _, bar_project_dir) = make_project(script.scratch_path, name='bar')
    (_, _, foo_project_dir) = make_project(script.scratch_path, name='foo', dependencies=[f'bar @ {path_to_url(str(bar_project_dir))}'])
    script.pip('install', '--config-settings', 'FOO=Hello', foo_project_dir)
    foo_config = script.site_packages_path / 'foo-config.json'
    with open(foo_config, 'rb') as f:
        assert json.load(f) == {'FOO': 'Hello'}
    bar_config = script.site_packages_path / 'bar-config.json'
    with open(bar_config, 'rb') as f:
        assert json.load(f) == {}

def test_dep_in_req_file_does_not_see_config(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Test that CLI config settings do not propagate to dependencies found in\n    requirement files.'
    (_, _, bar_project_dir) = make_project(script.scratch_path, name='bar')
    (_, _, foo_project_dir) = make_project(script.scratch_path, name='foo', dependencies=['bar'])
    reqs_file = script.scratch_path / 'reqs.txt'
    reqs_file.write_text(f'bar @ {path_to_url(str(bar_project_dir))}')
    script.pip('install', '--config-settings', 'FOO=Hello', '-r', reqs_file, foo_project_dir)
    foo_config = script.site_packages_path / 'foo-config.json'
    with open(foo_config, 'rb') as f:
        assert json.load(f) == {'FOO': 'Hello'}
    bar_config = script.site_packages_path / 'bar-config.json'
    with open(bar_config, 'rb') as f:
        assert json.load(f) == {}

def test_install_sees_config(script: PipTestEnvironment) -> None:
    if False:
        i = 10
        return i + 15
    (name, _, project_dir) = make_project(script.scratch_path)
    script.pip('install', '--config-settings', 'FOO=Hello', project_dir)
    config = script.site_packages_path / f'{name}-config.json'
    with open(config, 'rb') as f:
        assert json.load(f) == {'FOO': 'Hello'}

def test_install_sees_config_reqs(script: PipTestEnvironment) -> None:
    if False:
        i = 10
        return i + 15
    (name, _, project_dir) = make_project(script.scratch_path)
    script.scratch_path.joinpath('reqs.txt').write_text(f'{project_dir} --config-settings FOO=Hello')
    script.pip('install', '-r', 'reqs.txt')
    config = script.site_packages_path / f'{name}-config.json'
    with open(config, 'rb') as f:
        assert json.load(f) == {'FOO': 'Hello'}

def test_install_editable_sees_config(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    (name, _, project_dir) = make_project(script.scratch_path)
    script.pip('install', '--config-settings', 'FOO=Hello', '--editable', project_dir)
    config = script.site_packages_path / f'{name}-config.json'
    with open(config, 'rb') as f:
        assert json.load(f) == {'FOO': 'Hello'}

def test_install_config_reqs(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    (name, _, project_dir) = make_project(script.scratch_path)
    a_sdist = create_basic_sdist_for_package(script, 'foo', '1.0', {'pyproject.toml': PYPROJECT_TOML, 'backend/dummy_backend.py': BACKEND_SRC})
    script.scratch_path.joinpath('reqs.txt').write_text(f'{project_dir} --config-settings "--build-option=--cffi" --config-settings "--build-option=--avx2" --config-settings FOO=BAR')
    script.pip('install', '--no-index', '-f', str(a_sdist.parent), '-r', 'reqs.txt')
    script.assert_installed(foo='1.0')
    config = script.site_packages_path / f'{name}-config.json'
    with open(config, 'rb') as f:
        assert json.load(f) == {'--build-option': ['--cffi', '--avx2'], 'FOO': 'BAR'}