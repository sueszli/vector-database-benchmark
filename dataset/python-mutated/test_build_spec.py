import logging
import os
import sys
from unittest.mock import Mock
from lightning.app.testing import LightningTestApp, application_testing
from lightning.app.utilities.packaging.build_config import BuildConfig
from tests_app import _TESTS_ROOT
EXTRAS_ARGS = ['--blocking', 'False', '--multiprocess', '--open-ui', 'False']

class NoRequirementsLightningTestApp(LightningTestApp):

    def on_after_run_once(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.root.work.local_build_config.requirements == []
        assert self.root.work.cloud_build_config.requirements == []
        return super().on_after_run_once()

def test_build_config_no_requirements():
    if False:
        while True:
            i = 10
    command_line = [os.path.join(_TESTS_ROOT, 'utilities/packaging/projects/no_req/app.py')]
    application_testing(NoRequirementsLightningTestApp, command_line + EXTRAS_ARGS)
    sys.path = sys.path[:-1]

def test_build_config_requirements_provided():
    if False:
        i = 10
        return i + 15
    spec = BuildConfig(requirements=['dask', './projects/req/comp_req/a/requirements.txt'])
    assert spec.requirements == ['dask', 'pandas', 'pytorch_lightning==1.5.9', 'git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0']
    assert spec == BuildConfig.from_dict(spec.to_dict())

class BuildSpecTest(BuildConfig):

    def build_commands(self):
        if False:
            return 10
        return super().build_commands() + ['pip install redis']

def test_build_config_invalid_requirements():
    if False:
        for i in range(10):
            print('nop')
    spec = BuildSpecTest(requirements=['./projects/requirements.txt'])
    assert spec.requirements == ['cloud-stars']
    assert spec.build_commands() == ['pip install redis']

def test_build_config_dockerfile_provided():
    if False:
        while True:
            i = 10
    spec = BuildConfig(dockerfile='./projects/Dockerfile.cpu')
    assert not spec.requirements
    assert 'pytorchlightning/pytorch_lightning' in spec.dockerfile.data[0]

class DockerfileLightningTestApp(LightningTestApp):

    def on_after_run_once(self):
        if False:
            print('Hello World!')
        print(self.root.work.local_build_config.dockerfile)
        assert 'pytorchlightning/pytorch_' + 'lightning' in self.root.work.local_build_config.dockerfile.data[0]
        return super().on_after_run_once()

def test_build_config_dockerfile():
    if False:
        print('Hello World!')
    command_line = [os.path.join(_TESTS_ROOT, 'utilities/packaging/projects/dockerfile/app.py')]
    application_testing(DockerfileLightningTestApp, command_line + EXTRAS_ARGS)
    sys.path = sys.path[:-1]

class RequirementsLightningTestApp(LightningTestApp):

    def on_after_run_once(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.root.work.local_build_config.requirements == ['git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0', 'pandas', 'pytorch_' + 'lightning==1.5.9']
        return super().on_after_run_once()

def test_build_config_requirements():
    if False:
        for i in range(10):
            print('nop')
    command_line = [os.path.join(_TESTS_ROOT, 'utilities/packaging/projects/req/app.py')]
    application_testing(RequirementsLightningTestApp, command_line + EXTRAS_ARGS)
    sys.path = sys.path[:-1]

def test_build_config_requirements_warns(monkeypatch, caplog):
    if False:
        return 10
    requirements = ['foo', 'bar']
    bc = BuildConfig(requirements=requirements)
    monkeypatch.setattr(bc, '_find_requirements', lambda *_, **__: ['baz'])
    work = Mock()
    with caplog.at_level(logging.INFO):
        bc.on_work_init(work)
    assert "requirements.txt' exists with ['baz'] but ['foo', 'bar']" in caplog.text
    assert bc.requirements == requirements

def test_build_config_dockerfile_warns(monkeypatch, caplog):
    if False:
        while True:
            i = 10
    dockerfile = 'foo'
    bc = BuildConfig(dockerfile=dockerfile)
    monkeypatch.setattr(bc, '_find_dockerfile', lambda *_, **__: 'bar')
    work = Mock()
    with caplog.at_level(logging.INFO):
        bc.on_work_init(work)
    assert "exists at 'bar' but 'foo' was passed" in caplog.text
    assert bc.dockerfile == dockerfile