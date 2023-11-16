import os
import pathlib
import textwrap
from unittest import mock
import django_dynamic_fixture as fixture
import pytest
from django.conf import settings
from readthedocs.builds.constants import BUILD_STATUS_FAILURE, BUILD_STATUS_SUCCESS, EXTERNAL
from readthedocs.builds.models import Build
from readthedocs.config import ALL, ConfigError
from readthedocs.config.config import BuildConfigV2
from readthedocs.config.tests.test_config import get_build_config
from readthedocs.doc_builder.exceptions import BuildAppError
from readthedocs.projects.exceptions import RepositoryError
from readthedocs.projects.models import EnvironmentVariable, Project, WebHookEvent
from readthedocs.projects.tasks.builds import sync_repository_task, update_docs_task
from readthedocs.telemetry.models import BuildData
from .mockers import BuildEnvironmentMocker

@pytest.mark.django_db(databases='__all__')
class BuildEnvironmentBase:

    @pytest.fixture(autouse=True)
    def setup(self, requests_mock):
        if False:
            print('Hello World!')
        self.requests_mock = requests_mock
        self.project = self._get_project()
        self.version = self.project.versions.get(slug='latest')
        self.build = fixture.get(Build, version=self.version, commit='a1b2c3')
        self.mocker = BuildEnvironmentMocker(self.project, self.version, self.build, self.requests_mock)
        self.mocker.start()
        yield
        self.mocker.stop()

    def _get_project(self):
        if False:
            i = 10
            return i + 15
        return fixture.get(Project, slug='project', enable_epub_build=True, enable_pdf_build=True)

    def _trigger_update_docs_task(self):
        if False:
            while True:
                i = 10
        return update_docs_task.delay(self.version.pk, self.build.pk, build_api_key='1234', build_commit=self.build.commit)

class TestCustomConfigFile(BuildEnvironmentBase):
    config_file_name = 'unique.yaml'

    def _get_project(self):
        if False:
            print('Hello World!')
        return fixture.get(Project, slug='project', enable_epub_build=False, enable_pdf_build=False, readthedocs_yaml_path=self.config_file_name)

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    @mock.patch('readthedocs.doc_builder.director.BuildDirector.build_docs_class')
    def test_config_is_stored(self, build_docs_class, load_yaml_config):
        if False:
            i = 10
            return i + 15
        'Test that a custom config file is stored'
        config = get_build_config({'version': 2, 'formats': ['pdf'], 'sphinx': {'configuration': 'docs/conf.py'}}, source_file=self.config_file_name, validate=True)
        load_yaml_config.return_value = config
        build_docs_class.return_value = True
        assert not BuildData.objects.all().exists()
        self._trigger_update_docs_task()
        load_yaml_config.assert_called_once_with(version=mock.ANY, readthedocs_yaml_path=self.config_file_name)
        build_docs_class.assert_called_with('sphinx_pdf')

    @mock.patch('readthedocs.core.utils.filesystem._assert_path_is_inside_docroot')
    @mock.patch('readthedocs.doc_builder.director.BuildDirector.build_docs_class')
    def test_config_file_is_loaded(self, build_docs_class, _assert_path_is_inside_docroot):
        if False:
            return 10
        'Test that a custom config file is loaded\n\n        The readthedocs_yaml_path field on Project should be loading the file that we add\n        to the repo.'
        _assert_path_is_inside_docroot.return_value = True
        self.mocker.add_file_in_repo_checkout(self.config_file_name, textwrap.dedent('\n                version: 2\n                build:\n                  os: "ubuntu-22.04"\n                  tools:\n                    python: "3"\n                formats: [pdf]\n                sphinx:\n                  configuration: docs/conf.py\n        '))
        self._trigger_update_docs_task()
        build_docs_class.assert_called_with('sphinx_pdf')

class TestBuildTask(BuildEnvironmentBase):

    @pytest.mark.parametrize('formats,builders', ((['pdf'], ['latex']), (['htmlzip'], ['readthedocssinglehtmllocalmedia']), (['epub'], ['epub']), (['pdf', 'htmlzip', 'epub'], ['latex', 'readthedocssinglehtmllocalmedia', 'epub']), ('all', ['latex', 'readthedocssinglehtmllocalmedia', 'nepub'])))
    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    @pytest.mark.skip
    def test_build_sphinx_formats(self, load_yaml_config, formats, builders):
        if False:
            for i in range(10):
                print('nop')
        load_yaml_config.return_value = get_build_config({'version': 2, 'formats': formats, 'sphinx': {'configuration': 'docs/conf.py'}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_any_call(mock.call(mock.ANY, '-m', 'sphinx', '-T', '-E', '-b', 'html', '-d', '_build/doctrees', '-D', 'language=en', '.', '$READTHEDOCS_OUTPUT/html', cwd=mock.ANY, bin_path=mock.ANY))
        for builder in builders:
            self.mocker.mocks['environment.run'].assert_any_call(mock.call(mock.ANY, '-m', 'sphinx', '-T', '-E', '-b', builder, '-d', '_build/doctrees', '-D', 'language=en', '.', '$READTHEDOCS_OUTPUT/html', cwd=mock.ANY, bin_path=mock.ANY))

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    @mock.patch('readthedocs.doc_builder.director.BuildDirector.build_docs_class')
    def test_build_formats_only_html_for_external_versions(self, build_docs_class, load_yaml_config):
        if False:
            for i in range(10):
                print('nop')
        load_yaml_config.return_value = get_build_config({'version': 2, 'formats': 'all'}, validate=True)
        build_docs_class.return_value = True
        self.version.type = EXTERNAL
        self.version.save()
        self._trigger_update_docs_task()
        build_docs_class.assert_called_once_with('sphinx')

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    @mock.patch('readthedocs.doc_builder.director.BuildDirector.build_docs_class')
    def test_build_respects_formats_mkdocs(self, build_docs_class, load_yaml_config):
        if False:
            while True:
                i = 10
        load_yaml_config.return_value = get_build_config({'version': 2, 'mkdocs': {'configuration': 'mkdocs.yml'}, 'formats': ['epub', 'pdf']}, validate=True)
        self._trigger_update_docs_task()
        build_docs_class.assert_called_once_with('mkdocs')

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_build_updates_documentation_type(self, load_yaml_config):
        if False:
            return 10
        assert self.version.documentation_type == 'sphinx'
        load_yaml_config.return_value = get_build_config({'version': 2, 'mkdocs': {'configuration': 'mkdocs.yml'}, 'formats': ['epub', 'pdf']}, validate=True)
        os.makedirs(self.project.artifact_path(version=self.version.slug, type_='html'))
        for f in ('epub', 'pdf'):
            os.makedirs(self.project.artifact_path(version=self.version.slug, type_=f))
            pathlib.Path(os.path.join(self.project.artifact_path(version=self.version.slug, type_=f), f'{self.project.slug}.{f}')).touch()
        pathlib.Path(os.path.join(self.project.artifact_path(version=self.version.slug, type_='html'), 'index.html')).touch()
        self._trigger_update_docs_task()
        assert self.requests_mock.request_history[7]._request.method == 'PATCH'
        assert self.requests_mock.request_history[7].path == '/api/v2/version/1/'
        assert self.requests_mock.request_history[7].json() == {'addons': False, 'build_data': None, 'built': True, 'documentation_type': 'mkdocs', 'has_pdf': True, 'has_epub': True, 'has_htmlzip': False}

    @pytest.mark.parametrize('config', [{'version': 2}, {'version': 2, 'build': {'os': 'ubuntu-22.04', 'tools': {'python': '3.10'}, 'commands': ['echo Hello > index.html']}}])
    @pytest.mark.parametrize('external', [True, False])
    @mock.patch('readthedocs.projects.tasks.builds.LocalBuildEnvironment')
    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_get_env_vars(self, load_yaml_config, build_environment, config, external):
        if False:
            print('Hello World!')
        load_yaml_config.return_value = get_build_config(config, validate=True)
        if external:
            self.version.type = EXTERNAL
            self.version.save()
        fixture.get(EnvironmentVariable, name='PRIVATE_TOKEN', value='a1b2c3', project=self.project, public=False)
        fixture.get(EnvironmentVariable, name='PUBLIC_TOKEN', value='a1b2c3', project=self.project, public=True)
        common_env_vars = {'READTHEDOCS': 'True', 'READTHEDOCS_VERSION': self.version.slug, 'READTHEDOCS_VERSION_TYPE': self.version.type, 'READTHEDOCS_VERSION_NAME': self.version.verbose_name, 'READTHEDOCS_PROJECT': self.project.slug, 'READTHEDOCS_LANGUAGE': self.project.language, 'READTHEDOCS_OUTPUT': os.path.join(self.project.checkout_path(self.version.slug), '_readthedocs/'), 'READTHEDOCS_GIT_CLONE_URL': self.project.repo, 'READTHEDOCS_GIT_IDENTIFIER': self.version.identifier, 'READTHEDOCS_GIT_COMMIT_HASH': self.build.commit}
        self._trigger_update_docs_task()
        vcs_env_vars = build_environment.call_args_list[0][1]['environment']
        expected_vcs_env_vars = dict(**common_env_vars, GIT_TERMINAL_PROMPT='0')
        assert vcs_env_vars == expected_vcs_env_vars
        build_env_vars = build_environment.call_args_list[1][1]['environment']
        expected_build_env_vars = dict(**common_env_vars, NO_COLOR='1', BIN_PATH=os.path.join(self.project.doc_path, 'envs', self.version.slug, 'bin'), PUBLIC_TOKEN='a1b2c3', READTHEDOCS_VIRTUALENV_PATH=mock.ANY, READTHEDOCS_CANONICAL_URL=self.project.get_docs_url(lang_slug=self.project.language, version_slug=self.version.slug, external=external))
        if not external:
            expected_build_env_vars['PRIVATE_TOKEN'] = 'a1b2c3'
        assert build_env_vars == expected_build_env_vars

    @mock.patch('readthedocs.projects.tasks.builds.index_build')
    @mock.patch('readthedocs.projects.tasks.builds.build_complete')
    @mock.patch('readthedocs.projects.tasks.builds.send_external_build_status')
    @mock.patch('readthedocs.projects.tasks.builds.UpdateDocsTask.send_notifications')
    @mock.patch('readthedocs.projects.tasks.builds.clean_build')
    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_successful_build(self, load_yaml_config, clean_build, send_notifications, send_external_build_status, build_complete, index_build):
        if False:
            while True:
                i = 10
        load_yaml_config.return_value = get_build_config({'formats': 'all', 'sphinx': {'configuration': 'docs/conf.py'}}, validate=True)
        assert not BuildData.objects.all().exists()
        os.makedirs(self.project.artifact_path(version=self.version.slug, type_='html'))
        os.makedirs(self.project.artifact_path(version=self.version.slug, type_='json'))
        for f in ('htmlzip', 'epub', 'pdf'):
            os.makedirs(self.project.artifact_path(version=self.version.slug, type_=f))
            pathlib.Path(os.path.join(self.project.artifact_path(version=self.version.slug, type_=f), f'{self.project.slug}.{f}')).touch()
        pathlib.Path(os.path.join(self.project.artifact_path(version=self.version.slug, type_='html'), 'index.html')).touch()
        self._trigger_update_docs_task()
        clean_build.assert_has_calls([mock.call(mock.ANY), mock.call(mock.ANY)])
        send_notifications.assert_called_once_with(self.version.pk, self.build.pk, event=WebHookEvent.BUILD_PASSED)
        send_external_build_status.assert_called_once_with(version_type=self.version.type, build_pk=self.build.pk, commit=self.build.commit, status=BUILD_STATUS_SUCCESS)
        build_complete.send.assert_called_once_with(sender=Build, build=mock.ANY)
        index_build.delay.assert_called_once_with(build_id=self.build.pk)
        assert self.requests_mock.request_history[3].json() == {'id': 1, 'state': 'cloning', 'commit': 'a1b2c3', 'error': '', 'builder': mock.ANY}
        assert self.requests_mock.request_history[4].json() == {'id': 1, 'state': 'installing', 'commit': 'a1b2c3', 'builder': mock.ANY, 'readthedocs_yaml_path': None, 'error': '', 'config': {'version': '2', 'formats': ['htmlzip', 'pdf', 'epub'], 'python': {'install': []}, 'conda': None, 'build': {'os': 'ubuntu-22.04', 'commands': [], 'jobs': {'post_build': [], 'post_checkout': [], 'post_create_environment': [], 'post_install': [], 'post_system_dependencies': [], 'pre_build': [], 'pre_checkout': [], 'pre_create_environment': [], 'pre_install': [], 'pre_system_dependencies': []}, 'tools': {'python': {'full_version': '3.12.0', 'version': '3'}}, 'apt_packages': []}, 'doctype': 'sphinx', 'sphinx': {'builder': 'sphinx', 'configuration': 'docs/conf.py', 'fail_on_warning': False}, 'mkdocs': None, 'submodules': {'include': [], 'exclude': 'all', 'recursive': False}, 'search': {'ranking': {}, 'ignore': ['search.html', 'search/index.html', '404.html', '404/index.html']}}}
        assert self.requests_mock.request_history[5].json() == {'id': 1, 'state': 'building', 'commit': 'a1b2c3', 'readthedocs_yaml_path': None, 'config': mock.ANY, 'builder': mock.ANY, 'error': ''}
        assert self.requests_mock.request_history[6].json() == {'id': 1, 'state': 'uploading', 'commit': 'a1b2c3', 'readthedocs_yaml_path': None, 'config': mock.ANY, 'builder': mock.ANY, 'error': ''}
        assert self.requests_mock.request_history[7]._request.method == 'PATCH'
        assert self.requests_mock.request_history[7].path == '/api/v2/version/1/'
        assert self.requests_mock.request_history[7].json() == {'addons': False, 'build_data': None, 'built': True, 'documentation_type': 'sphinx', 'has_pdf': True, 'has_epub': True, 'has_htmlzip': True}
        assert self.requests_mock.request_history[8]._request.method == 'PATCH'
        assert self.requests_mock.request_history[8].path == '/api/v2/project/1/'
        assert self.requests_mock.request_history[8].json() == {'has_valid_clone': True}
        assert self.requests_mock.request_history[9].json() == {'id': 1, 'state': 'finished', 'commit': 'a1b2c3', 'readthedocs_yaml_path': None, 'config': mock.ANY, 'builder': mock.ANY, 'length': mock.ANY, 'success': True, 'error': ''}
        assert self.requests_mock.request_history[10]._request.method == 'POST'
        assert self.requests_mock.request_history[10].path == '/api/v2/revoke/'
        assert BuildData.objects.all().exists()
        self.mocker.mocks['build_media_storage'].rclone_sync_directory.assert_has_calls([mock.call(mock.ANY, 'html/project/latest'), mock.call(mock.ANY, 'json/project/latest'), mock.call(mock.ANY, 'htmlzip/project/latest'), mock.call(mock.ANY, 'pdf/project/latest'), mock.call(mock.ANY, 'epub/project/latest')])

    @mock.patch('readthedocs.projects.tasks.builds.build_complete')
    @mock.patch('readthedocs.projects.tasks.builds.send_external_build_status')
    @mock.patch('readthedocs.projects.tasks.builds.UpdateDocsTask.execute')
    @mock.patch('readthedocs.projects.tasks.builds.UpdateDocsTask.send_notifications')
    @mock.patch('readthedocs.projects.tasks.builds.clean_build')
    def test_failed_build(self, clean_build, send_notifications, execute, send_external_build_status, build_complete):
        if False:
            return 10
        assert not BuildData.objects.all().exists()
        execute.side_effect = Exception('Force and exception here.')
        self._trigger_update_docs_task()
        clean_build.assert_has_calls([mock.call(mock.ANY), mock.call(mock.ANY)])
        send_notifications.assert_called_once_with(self.version.pk, self.build.pk, event=WebHookEvent.BUILD_FAILED)
        send_external_build_status.assert_called_once_with(version_type=self.version.type, build_pk=self.build.pk, commit=self.build.commit, status=BUILD_STATUS_FAILURE)
        build_complete.send.assert_called_once_with(sender=Build, build=mock.ANY)
        assert not BuildData.objects.all().exists()
        api_request = self.requests_mock.request_history[-2]
        assert api_request._request.method == 'PATCH'
        assert api_request.path == '/api/v2/build/1/'
        assert api_request.json() == {'builder': mock.ANY, 'commit': self.build.commit, 'error': BuildAppError.GENERIC_WITH_BUILD_ID.format(build_id=self.build.pk), 'id': self.build.pk, 'length': mock.ANY, 'state': 'finished', 'success': False}
        revoke_key_request = self.requests_mock.request_history[-1]
        assert revoke_key_request._request.method == 'POST'
        assert revoke_key_request.path == '/api/v2/revoke/'

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_build_commands_executed(self, load_yaml_config):
        if False:
            i = 10
            return i + 15
        load_yaml_config.return_value = get_build_config({'version': 2, 'formats': 'all', 'sphinx': {'configuration': 'docs/conf.py'}}, validate=True)
        os.makedirs(self.project.artifact_path(version=self.version.slug, type_='html'))
        os.makedirs(self.project.artifact_path(version=self.version.slug, type_='json'))
        os.makedirs(self.project.artifact_path(version=self.version.slug, type_='htmlzip'))
        os.makedirs(self.project.artifact_path(version=self.version.slug, type_='epub'))
        os.makedirs(self.project.artifact_path(version=self.version.slug, type_='pdf'))
        self._trigger_update_docs_task()
        self.mocker.mocks['git.Backend.run'].assert_has_calls([mock.call('git', 'clone', '--depth', '1', mock.ANY, '.'), mock.call('git', 'fetch', 'origin', '--force', '--prune', '--prune-tags', '--depth', '50'), mock.call('git', 'show-ref', '--verify', '--quiet', '--', 'refs/remotes/origin/a1b2c3', record=False), mock.call('git', 'checkout', '--force', 'origin/a1b2c3'), mock.call('git', 'clean', '-d', '-f', '-f'), mock.call('git', 'ls-remote', '--tags', '--heads', mock.ANY, demux=True, record=False)])
        python_version = settings.RTD_DOCKER_BUILD_SETTINGS['tools']['python']['3']
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call('cat', 'readthedocs.yml', cwd='/tmp/readthedocs-tests/git-repository'), mock.call('asdf', 'install', 'python', python_version), mock.call('asdf', 'global', 'python', python_version), mock.call('asdf', 'reshim', 'python', record=False), mock.call('python', '-mpip', 'install', '-U', 'virtualenv', 'setuptools'), mock.call('python', '-mvirtualenv', '$READTHEDOCS_VIRTUALENV_PATH', bin_path=None, cwd=None), mock.call(mock.ANY, '-m', 'pip', 'install', '--upgrade', '--no-cache-dir', 'pip', 'setuptools', bin_path=mock.ANY, cwd=mock.ANY), mock.call(mock.ANY, '-m', 'pip', 'install', '--upgrade', '--no-cache-dir', 'sphinx', 'readthedocs-sphinx-ext', bin_path=mock.ANY, cwd=mock.ANY), mock.call(mock.ANY, '-m', 'sphinx', '-T', '-E', '-b', 'html', '-d', '_build/doctrees', '-D', 'language=en', '.', '$READTHEDOCS_OUTPUT/html', cwd=mock.ANY, bin_path=mock.ANY), mock.call(mock.ANY, '-m', 'sphinx', '-T', '-E', '-b', 'readthedocssinglehtmllocalmedia', '-d', '_build/doctrees', '-D', 'language=en', '.', '$READTHEDOCS_OUTPUT/htmlzip', cwd=mock.ANY, bin_path=mock.ANY), mock.call('mktemp', '--directory', record=False), mock.call('mv', mock.ANY, mock.ANY, cwd=mock.ANY, record=False), mock.call('mkdir', '--parents', mock.ANY, cwd=mock.ANY, record=False), mock.call('zip', '--recurse-paths', '--symlinks', mock.ANY, mock.ANY, cwd=mock.ANY, record=False), mock.call(mock.ANY, '-m', 'sphinx', '-T', '-E', '-b', 'latex', '-d', '_build/doctrees', '-D', 'language=en', '.', '$READTHEDOCS_OUTPUT/pdf', cwd=mock.ANY, bin_path=mock.ANY), mock.call('cat', 'latexmkrc', cwd=mock.ANY), mock.call(mock.ANY, '-m', 'sphinx', '-T', '-E', '-b', 'epub', '-d', '_build/doctrees', '-D', 'language=en', '.', '$READTHEDOCS_OUTPUT/epub', cwd=mock.ANY, bin_path=mock.ANY), mock.call('mv', mock.ANY, '/tmp/project-latest.epub', cwd=mock.ANY, record=False), mock.call('rm', '--recursive', '$READTHEDOCS_OUTPUT/epub', cwd=mock.ANY, record=False), mock.call('mkdir', '--parents', '$READTHEDOCS_OUTPUT/epub', cwd=mock.ANY, record=False), mock.call('mv', '/tmp/project-latest.epub', mock.ANY, cwd=mock.ANY, record=False), mock.call('test', '-x', '_build/html', record=False, cwd=mock.ANY), mock.call('lsb_release', '--description', record=False, demux=True), mock.call('python', '--version', record=False, demux=True), mock.call('dpkg-query', '--showformat', '${package} ${version}\\n', '--show', record=False, demux=True), mock.call('python', '-m', 'pip', 'list', '--pre', '--local', '--format', 'json', record=False, demux=True)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_install_apt_packages(self, load_yaml_config):
        if False:
            print('Hello World!')
        config = BuildConfigV2({'version': 2, 'build': {'os': 'ubuntu-22.04', 'tools': {'python': '3'}, 'apt_packages': ['clangd', 'cmatrix']}}, source_file='readthedocs.yml')
        config.validate()
        load_yaml_config.return_value = config
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call('apt-get', 'update', '--assume-yes', '--quiet', user='root:root'), mock.call('apt-get', 'install', '--assume-yes', '--quiet', '--', 'clangd', 'cmatrix', user='root:root')])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_build_tools(self, load_yaml_config):
        if False:
            i = 10
            return i + 15
        config = BuildConfigV2({'version': 2, 'build': {'os': 'ubuntu-20.04', 'tools': {'python': '3.10', 'nodejs': '16', 'rust': '1.55', 'golang': '1.17'}}}, source_file='readthedocs.yml')
        config.validate()
        load_yaml_config.return_value = config
        self._trigger_update_docs_task()
        python_version = settings.RTD_DOCKER_BUILD_SETTINGS['tools']['python']['3.10']
        nodejs_version = settings.RTD_DOCKER_BUILD_SETTINGS['tools']['nodejs']['16']
        rust_version = settings.RTD_DOCKER_BUILD_SETTINGS['tools']['rust']['1.55']
        golang_version = settings.RTD_DOCKER_BUILD_SETTINGS['tools']['golang']['1.17']
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call('asdf', 'install', 'python', python_version), mock.call('asdf', 'global', 'python', python_version), mock.call('asdf', 'reshim', 'python', record=False), mock.call('python', '-mpip', 'install', '-U', 'virtualenv', 'setuptools'), mock.call('asdf', 'install', 'nodejs', nodejs_version), mock.call('asdf', 'global', 'nodejs', nodejs_version), mock.call('asdf', 'reshim', 'nodejs', record=False), mock.call('asdf', 'install', 'rust', rust_version), mock.call('asdf', 'global', 'rust', rust_version), mock.call('asdf', 'reshim', 'rust', record=False), mock.call('asdf', 'install', 'golang', golang_version), mock.call('asdf', 'global', 'golang', golang_version), mock.call('asdf', 'reshim', 'golang', record=False), mock.ANY])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_build_jobs(self, load_yaml_config):
        if False:
            i = 10
            return i + 15
        config = BuildConfigV2({'version': 2, 'build': {'os': 'ubuntu-20.04', 'tools': {'python': '3.7'}, 'jobs': {'post_checkout': ['git fetch --unshallow'], 'pre_build': ['echo `date`']}}}, source_file='readthedocs.yml')
        config.validate()
        load_yaml_config.return_value = config
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call('git fetch --unshallow', escape_command=False, cwd=mock.ANY), mock.call('echo `date`', escape_command=False, cwd=mock.ANY)], any_order=True)

    @mock.patch('readthedocs.doc_builder.director.tarfile')
    @mock.patch('readthedocs.doc_builder.director.build_tools_storage')
    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_build_tools_cached(self, load_yaml_config, build_tools_storage, tarfile):
        if False:
            for i in range(10):
                print('nop')
        config = BuildConfigV2({'version': 2, 'build': {'os': 'ubuntu-20.04', 'tools': {'python': '3.10', 'nodejs': '16', 'rust': '1.55', 'golang': '1.17'}}}, source_file='readthedocs.yml')
        config.validate()
        load_yaml_config.return_value = config
        build_tools_storage.open.return_value = b''
        build_tools_storage.exists.return_value = True
        tarfile.open.return_value.__enter__.return_value.extract_all.return_value = None
        self._trigger_update_docs_task()
        python_version = settings.RTD_DOCKER_BUILD_SETTINGS['tools']['python']['3.10']
        nodejs_version = settings.RTD_DOCKER_BUILD_SETTINGS['tools']['nodejs']['16']
        rust_version = settings.RTD_DOCKER_BUILD_SETTINGS['tools']['rust']['1.55']
        golang_version = settings.RTD_DOCKER_BUILD_SETTINGS['tools']['golang']['1.17']
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call('mv', mock.ANY, f'/home/docs/.asdf/installs/python/{python_version}', record=False), mock.call('asdf', 'global', 'python', python_version), mock.call('asdf', 'reshim', 'python', record=False), mock.call('mv', mock.ANY, f'/home/docs/.asdf/installs/nodejs/{nodejs_version}', record=False), mock.call('asdf', 'global', 'nodejs', nodejs_version), mock.call('asdf', 'reshim', 'nodejs', record=False), mock.call('mv', mock.ANY, f'/home/docs/.asdf/installs/rust/{rust_version}', record=False), mock.call('asdf', 'global', 'rust', rust_version), mock.call('asdf', 'reshim', 'rust', record=False), mock.call('mv', mock.ANY, f'/home/docs/.asdf/installs/golang/{golang_version}', record=False), mock.call('asdf', 'global', 'golang', golang_version), mock.call('asdf', 'reshim', 'golang', record=False), mock.ANY])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_build_commands(self, load_yaml_config):
        if False:
            for i in range(10):
                print('nop')
        config = BuildConfigV2({'version': 2, 'build': {'os': 'ubuntu-22.04', 'tools': {'python': '3.10'}, 'commands': ['pip install pelican[markdown]', 'pelican --settings docs/pelicanconf.py --output $READTHEDOCS_OUTPUT/html/ docs/']}}, source_file='readthedocs.yml')
        config.validate()
        load_yaml_config.return_value = config
        self._trigger_update_docs_task()
        python_version = settings.RTD_DOCKER_BUILD_SETTINGS['tools']['python']['3.10']
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call('asdf', 'install', 'python', python_version), mock.call('asdf', 'global', 'python', python_version), mock.call('asdf', 'reshim', 'python', record=False), mock.call('python', '-mpip', 'install', '-U', 'virtualenv', 'setuptools'), mock.call('pip install pelican[markdown]', escape_command=False, cwd=mock.ANY), mock.call('asdf', 'reshim', 'python', escape_command=False, record=False, cwd=mock.ANY), mock.call('pelican --settings docs/pelicanconf.py --output $READTHEDOCS_OUTPUT/html/ docs/', escape_command=False, cwd=mock.ANY)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_requirements_from_config_file_installed(self, load_yaml_config):
        if False:
            for i in range(10):
                print('nop')
        load_yaml_config.return_value = get_build_config({'version': 2, 'python': {'install': [{'requirements': 'requirements.txt'}]}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call(mock.ANY, '-m', 'pip', 'install', '--exists-action=w', '--no-cache-dir', '-r', 'requirements.txt', cwd=mock.ANY, bin_path=mock.ANY)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_conda_config_calls_conda_command(self, load_yaml_config):
        if False:
            for i in range(10):
                print('nop')
        load_yaml_config.return_value = get_build_config({'version': 2, 'build': {'os': 'ubuntu-22.04', 'tools': {'python': 'miniconda3-4.7'}}, 'conda': {'environment': 'environment.yaml'}}, validate=True)
        self._trigger_update_docs_task()
        python_version = settings.RTD_DOCKER_BUILD_SETTINGS['tools']['python']['miniconda3-4.7']
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call('cat', 'readthedocs.yml', cwd=mock.ANY), mock.call('asdf', 'install', 'python', python_version), mock.call('asdf', 'global', 'python', python_version), mock.call('asdf', 'reshim', 'python', record=False), mock.call('conda', 'env', 'create', '--quiet', '--name', self.version.slug, '--file', 'environment.yaml', cwd=mock.ANY, bin_path=mock.ANY), mock.call('conda', 'install', '--yes', '--quiet', '--name', self.version.slug, 'sphinx', cwd=mock.ANY), mock.call(mock.ANY, '-m', 'pip', 'install', '-U', '--no-cache-dir', 'readthedocs-sphinx-ext', cwd=mock.ANY, bin_path=mock.ANY), mock.call('test', '-x', '_build/html', cwd=mock.ANY, record=False), mock.call('lsb_release', '--description', record=False, demux=True), mock.call('python', '--version', record=False, demux=True), mock.call('dpkg-query', '--showformat', '${package} ${version}\\n', '--show', record=False, demux=True), mock.call('conda', 'list', '--json', '--name', 'latest', record=False, demux=True), mock.call('python', '-m', 'pip', 'list', '--pre', '--local', '--format', 'json', record=False, demux=True)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_python_mamba_commands(self, load_yaml_config):
        if False:
            i = 10
            return i + 15
        load_yaml_config.return_value = get_build_config({'version': 2, 'build': {'os': 'ubuntu-20.04', 'tools': {'python': 'mambaforge-4.10'}}, 'conda': {'environment': 'environment.yaml'}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call('cat', 'readthedocs.yml', cwd=mock.ANY), mock.call('asdf', 'install', 'python', 'mambaforge-4.10.3-10'), mock.call('asdf', 'global', 'python', 'mambaforge-4.10.3-10'), mock.call('asdf', 'reshim', 'python', record=False), mock.call('mamba', 'env', 'create', '--quiet', '--name', 'latest', '--file', 'environment.yaml', bin_path=None, cwd=mock.ANY), mock.call('mamba', 'install', '--yes', '--quiet', '--name', 'latest', 'sphinx', cwd=mock.ANY), mock.call(mock.ANY, '-m', 'pip', 'install', '-U', '--no-cache-dir', 'readthedocs-sphinx-ext', bin_path=mock.ANY, cwd=mock.ANY)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_sphinx_normalized_language(self, load_yaml_config):
        if False:
            return 10
        load_yaml_config.return_value = get_build_config({'sphinx': {'configuration': 'docs/conf.py', 'fail_on_warning': True}}, validate=True)
        self.project.language = 'es-mx'
        self.project.save()
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call(mock.ANY, '-m', 'sphinx', '-T', '-E', '-W', '--keep-going', '-b', 'html', '-d', '_build/doctrees', '-D', 'language=es_MX', '.', '$READTHEDOCS_OUTPUT/html', cwd=mock.ANY, bin_path=mock.ANY)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_sphinx_fail_on_warning(self, load_yaml_config):
        if False:
            i = 10
            return i + 15
        load_yaml_config.return_value = get_build_config({'version': 2, 'sphinx': {'configuration': 'docs/conf.py', 'fail_on_warning': True}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call(mock.ANY, '-m', 'sphinx', '-T', '-E', '-W', '--keep-going', '-b', 'html', '-d', '_build/doctrees', '-D', 'language=en', '.', '$READTHEDOCS_OUTPUT/html', cwd=mock.ANY, bin_path=mock.ANY)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_mkdocs_fail_on_warning(self, load_yaml_config):
        if False:
            print('Hello World!')
        load_yaml_config.return_value = get_build_config({'version': 2, 'mkdocs': {'configuration': 'docs/mkdocs.yaml', 'fail_on_warning': True}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call(mock.ANY, '-m', 'mkdocs', 'build', '--clean', '--site-dir', '$READTHEDOCS_OUTPUT/html', '--config-file', 'docs/mkdocs.yaml', '--strict', cwd=mock.ANY, bin_path=mock.ANY)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_python_install_setuptools(self, load_yaml_config):
        if False:
            for i in range(10):
                print('nop')
        load_yaml_config.return_value = get_build_config({'version': 2, 'python': {'install': [{'path': '.', 'method': 'setuptools'}]}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call(mock.ANY, './setup.py', 'install', '--force', cwd=mock.ANY, bin_path=mock.ANY)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_python_install_pip(self, load_yaml_config):
        if False:
            for i in range(10):
                print('nop')
        load_yaml_config.return_value = get_build_config({'version': 2, 'python': {'install': [{'path': '.', 'method': 'pip'}]}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call(mock.ANY, '-m', 'pip', 'install', '--upgrade', '--upgrade-strategy', 'only-if-needed', '--no-cache-dir', '.', cwd=mock.ANY, bin_path=mock.ANY)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_python_install_pip_extras(self, load_yaml_config):
        if False:
            while True:
                i = 10
        load_yaml_config.return_value = get_build_config({'version': 2, 'python': {'install': [{'path': '.', 'method': 'pip', 'extra_requirements': ['docs']}]}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call(mock.ANY, '-m', 'pip', 'install', '--upgrade', '--upgrade-strategy', 'only-if-needed', '--no-cache-dir', '.[docs]', cwd=mock.ANY, bin_path=mock.ANY)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_python_install_pip_several_options(self, load_yaml_config):
        if False:
            print('Hello World!')
        load_yaml_config.return_value = get_build_config({'version': 2, 'python': {'install': [{'path': '.', 'method': 'pip', 'extra_requirements': ['docs']}, {'path': 'two', 'method': 'setuptools'}, {'requirements': 'three.txt'}]}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call(mock.ANY, '-m', 'pip', 'install', '--upgrade', '--upgrade-strategy', 'only-if-needed', '--no-cache-dir', '.[docs]', cwd=mock.ANY, bin_path=mock.ANY), mock.call(mock.ANY, 'two/setup.py', 'install', '--force', cwd=mock.ANY, bin_path=mock.ANY), mock.call(mock.ANY, '-m', 'pip', 'install', '--exists-action=w', '--no-cache-dir', '-r', 'three.txt', cwd=mock.ANY, bin_path=mock.ANY)])

    @pytest.mark.parametrize('value,expected', [(ALL, []), (['one', 'two'], ['one', 'two'])])
    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_submodules_include(self, load_yaml_config, value, expected):
        if False:
            while True:
                i = 10
        load_yaml_config.return_value = get_build_config({'version': 2, 'submodules': {'include': value}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['git.Backend.run'].assert_has_calls([mock.call('git', 'submodule', 'sync'), mock.call('git', 'submodule', 'update', '--init', '--force', '--', *expected)])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_submodules_exclude(self, load_yaml_config):
        if False:
            return 10
        load_yaml_config.return_value = get_build_config({'version': 2, 'submodules': {'exclude': ['one'], 'recursive': True}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['git.Backend.run'].assert_has_calls([mock.call('git', 'submodule', 'sync'), mock.call('git', 'submodule', 'update', '--init', '--force', '--recursive', '--', 'two', 'three')])

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_submodules_exclude_all(self, load_yaml_config):
        if False:
            while True:
                i = 10
        load_yaml_config.return_value = get_build_config({'version': 2, 'submodules': {'exclude': ALL, 'recursive': True}}, validate=True)
        self._trigger_update_docs_task()
        for call in self.mocker.mocks['git.Backend.run'].mock_calls:
            if 'submodule' in call.args:
                assert False, 'git submodule command found'

    @pytest.mark.parametrize('value,command', [('html', 'html'), ('htmldir', 'dirhtml'), ('dirhtml', 'dirhtml'), ('singlehtml', 'singlehtml')])
    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_sphinx_builder(self, load_yaml_config, value, command):
        if False:
            for i in range(10):
                print('nop')
        load_yaml_config.return_value = get_build_config({'version': 2, 'sphinx': {'builder': value, 'configuration': 'docs/conf.py'}}, validate=True)
        self._trigger_update_docs_task()
        self.mocker.mocks['environment.run'].assert_has_calls([mock.call(mock.ANY, '-m', 'sphinx', '-T', '-E', '-b', command, '-d', '_build/doctrees', '-D', 'language=en', '.', '$READTHEDOCS_OUTPUT/html', cwd=mock.ANY, bin_path=mock.ANY)])

class TestBuildTaskExceptionHandler(BuildEnvironmentBase):

    @mock.patch('readthedocs.doc_builder.director.load_yaml_config')
    def test_config_file_exception(self, load_yaml_config):
        if False:
            print('Hello World!')
        load_yaml_config.side_effect = ConfigError(code='invalid', message='Invalid version in config file.')
        self._trigger_update_docs_task()
        api_request = self.requests_mock.request_history[-2]
        assert api_request._request.method == 'PATCH'
        assert api_request.path == '/api/v2/build/1/'
        assert api_request.json() == {'id': 1, 'state': 'finished', 'commit': 'a1b2c3', 'error': "Problem in your project's configuration. Invalid version in config file.", 'success': False, 'builder': mock.ANY, 'length': 0}
        revoke_key_request = self.requests_mock.request_history[-1]
        assert revoke_key_request._request.method == 'POST'
        assert revoke_key_request.path == '/api/v2/revoke/'

class TestSyncRepositoryTask(BuildEnvironmentBase):

    def _trigger_sync_repository_task(self):
        if False:
            for i in range(10):
                print('nop')
        sync_repository_task.delay(self.version.pk, build_api_key='1234')

    @mock.patch('readthedocs.projects.tasks.builds.clean_build')
    def test_clean_build_after_sync_repository(self, clean_build):
        if False:
            i = 10
            return i + 15
        self._trigger_sync_repository_task()
        clean_build.assert_called_once()

    @mock.patch('readthedocs.projects.tasks.builds.SyncRepositoryTask.execute')
    @mock.patch('readthedocs.projects.tasks.builds.clean_build')
    def test_clean_build_after_failure_in_sync_repository(self, clean_build, execute):
        if False:
            i = 10
            return i + 15
        execute.side_effect = Exception('Something weird happen')
        self._trigger_sync_repository_task()
        clean_build.assert_called_once()

    @pytest.mark.parametrize('verbose_name', ['stable', 'latest'])
    @mock.patch('readthedocs.projects.tasks.builds.SyncRepositoryTask.on_failure')
    def test_check_duplicate_reserved_version_latest(self, on_failure, verbose_name):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('readthedocs.vcs_support.backends.git.Backend.lsremote', return_value=[[mock.MagicMock(identifier='branch/a1b2c3', verbose_name=verbose_name)], [mock.MagicMock(identifier='tag/a1b2c3', verbose_name=verbose_name)]]):
            self._trigger_sync_repository_task()
        on_failure.assert_called_once_with(mock.ANY, mock.ANY, [self.version.pk], {'build_api_key': mock.ANY}, mock.ANY)
        exception = on_failure.call_args[0][0]
        assert isinstance(exception, RepositoryError) == True
        assert exception.message == RepositoryError.DUPLICATED_RESERVED_VERSIONS