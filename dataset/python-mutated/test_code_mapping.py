from unittest.mock import patch
import pytest
from sentry.integrations.utils.code_mapping import CodeMapping, CodeMappingTreesHelper, FrameFilename, Repo, RepoTree, UnsupportedFrameFilename, filter_source_code_files, get_extension, get_sorted_code_mapping_configs, should_include, stacktrace_buckets
from sentry.models.integrations.integration import Integration
from sentry.models.integrations.organization_integration import OrganizationIntegration
from sentry.silo.base import SiloMode
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import assume_test_silo_mode
sentry_files = ['bin/__init__.py', 'bin/example1.py', 'bin/example2.py', 'docs-ui/.eslintrc.js', 'src/sentry/identity/oauth2.py', 'src/sentry/integrations/slack/client.py', 'src/sentry/web/urls.py', 'src/sentry/wsgi.py', 'src/sentry_plugins/slack/client.py']
UNSUPPORTED_FRAME_FILENAMES = ['async https://s1.sentry-cdn.com/_static/dist/sentry/entrypoints/app.js', '/gtm.js', '<anonymous>', '<frozen importlib._bootstrap>', '[native code]', 'O$t', 'async https://s1.sentry-cdn.com/_static/dist/sentry/entrypoints/app.js', '/foo/bar/baz', 'README', 'ssl.py', 'C:\\Users\\Donia\\AppData\\Roaming\\Adobe\\UXP\\Plugins\\External\\452f92d2_0.13.0\\main.js', 'initialization.dart', 'backburner.js']

class TestRepoFiles(TestCase):
    """These evaluate which files should be included as part of a repo."""

    def test_filter_source_code_files(self):
        if False:
            i = 10
            return i + 15
        source_code_files = filter_source_code_files(sentry_files)
        assert source_code_files.index('bin/__init__.py') == 0
        assert source_code_files.index('docs-ui/.eslintrc.js') == 3
        with pytest.raises(ValueError):
            source_code_files.index('README.md')

    def test_filter_source_code_files_not_supported(self):
        if False:
            return 10
        source_code_files = filter_source_code_files([])
        assert source_code_files == []
        source_code_files = filter_source_code_files(['.env', 'README'])
        assert source_code_files == []

    def test_should_not_include(self):
        if False:
            for i in range(10):
                print('nop')
        for file in ['static/app/views/organizationRoot.spec.jsx', 'tests/foo.py']:
            assert should_include(file) is False

def test_get_extension():
    if False:
        i = 10
        return i + 15
    assert get_extension('') == ''
    assert get_extension('f.py') == 'py'
    assert get_extension('f.xx') == 'xx'
    assert get_extension('./app/utils/handleXhrErrorResponse.tsx') == 'tsx'
    assert get_extension('[native code]') == ''
    assert get_extension('/foo/bar/baz') == ''
    assert get_extension('/gtm.js') == 'js'

def test_buckets_logic():
    if False:
        for i in range(10):
            print('nop')
    stacktraces = ['app://foo.js', './app/utils/handleXhrErrorResponse.tsx', 'getsentry/billing/tax/manager.py', '/cronscripts/monitoringsync.php'] + UNSUPPORTED_FRAME_FILENAMES
    buckets = stacktrace_buckets(stacktraces)
    assert buckets == {'./app': [FrameFilename('./app/utils/handleXhrErrorResponse.tsx')], 'app:': [FrameFilename('app://foo.js')], 'cronscripts': [FrameFilename('/cronscripts/monitoringsync.php')], 'getsentry': [FrameFilename('getsentry/billing/tax/manager.py')]}

class TestFrameFilename:

    def test_frame_filename_package_and_more_than_one_level(self):
        if False:
            while True:
                i = 10
        ff = FrameFilename('getsentry/billing/tax/manager.py')
        assert f'{ff.root}/{ff.dir_path}/{ff.file_name}' == 'getsentry/billing/tax/manager.py'
        assert f'{ff.dir_path}/{ff.file_name}' == ff.file_and_dir_path

    def test_frame_filename_package_and_no_levels(self):
        if False:
            while True:
                i = 10
        ff = FrameFilename('root/bar.py')
        assert f'{ff.root}/{ff.file_name}' == 'root/bar.py'
        assert f'{ff.root}/{ff.file_and_dir_path}' == 'root/bar.py'
        assert ff.dir_path == ''

    def test_frame_filename_repr(self):
        if False:
            for i in range(10):
                print('nop')
        path = 'getsentry/billing/tax/manager.py'
        assert FrameFilename(path).__repr__() == f'FrameFilename: {path}'

    def test_raises_unsupported(self):
        if False:
            print('Hello World!')
        for filepath in UNSUPPORTED_FRAME_FILENAMES:
            with pytest.raises(UnsupportedFrameFilename):
                FrameFilename(filepath)

class TestDerivedCodeMappings(TestCase):

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        if False:
            i = 10
            return i + 15
        self._caplog = caplog

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.foo_repo = Repo('Test-Organization/foo', 'master')
        self.bar_repo = Repo('Test-Organization/bar', 'main')
        self.code_mapping_helper = CodeMappingTreesHelper({self.foo_repo.name: RepoTree(self.foo_repo, files=sentry_files), self.bar_repo.name: RepoTree(self.bar_repo, files=['sentry/web/urls.py'])})
        self.expected_code_mappings = [CodeMapping(repo=self.foo_repo, stacktrace_root='sentry/', source_path='src/sentry/'), CodeMapping(repo=self.foo_repo, stacktrace_root='sentry_plugins/', source_path='src/sentry_plugins/')]

    def test_package_also_matches(self):
        if False:
            while True:
                i = 10
        repo_tree = RepoTree(self.foo_repo, files=['apostello/views/base.py'])
        cmh = CodeMappingTreesHelper({self.foo_repo.name: repo_tree})
        cm = cmh._generate_code_mapping_from_tree(repo_tree=repo_tree, frame_filename=FrameFilename('raven/base.py'))
        assert cm == []

    def test_no_matches(self):
        if False:
            for i in range(10):
                print('nop')
        stacktraces = ['getsentry/billing/tax/manager.py', 'requests/models.py', 'urllib3/connectionpool.py', 'ssl.py']
        code_mappings = self.code_mapping_helper.generate_code_mappings(stacktraces)
        assert code_mappings == []

    @patch('sentry.integrations.utils.code_mapping.logger')
    def test_matches_top_src_file(self, logger):
        if False:
            while True:
                i = 10
        stacktraces = ['setup.py']
        code_mappings = self.code_mapping_helper.generate_code_mappings(stacktraces)
        assert code_mappings == []

    def test_no_dir_depth_match(self):
        if False:
            print('Hello World!')
        code_mappings = self.code_mapping_helper.generate_code_mappings(['sentry/wsgi.py'])
        assert code_mappings == [CodeMapping(repo=Repo(name='Test-Organization/foo', branch='master'), stacktrace_root='sentry/', source_path='src/sentry/')]

    def test_more_than_one_match_does_derive(self):
        if False:
            return 10
        stacktraces = ['sentry_plugins/slack/client.py']
        code_mappings = self.code_mapping_helper.generate_code_mappings(stacktraces)
        assert code_mappings == [CodeMapping(repo=self.foo_repo, stacktrace_root='sentry_plugins/', source_path='src/sentry_plugins/')]

    def test_no_stacktraces_to_process(self):
        if False:
            while True:
                i = 10
        code_mappings = self.code_mapping_helper.generate_code_mappings([])
        assert code_mappings == []

    def test_more_than_one_match_works_when_code_mapping_excludes_other_match(self):
        if False:
            print('Hello World!')
        stacktraces = ['sentry/identity/oauth2.py', 'sentry_plugins/slack/client.py']
        code_mappings = self.code_mapping_helper.generate_code_mappings(stacktraces)
        assert code_mappings == self.expected_code_mappings

    def test_more_than_one_match_works_with_different_order(self):
        if False:
            return 10
        stacktraces = ['sentry_plugins/slack/client.py', 'sentry/identity/oauth2.py']
        code_mappings = self.code_mapping_helper.generate_code_mappings(stacktraces)
        assert sorted(code_mappings) == sorted(self.expected_code_mappings)

    @patch('sentry.integrations.utils.code_mapping.logger')
    def test_more_than_one_repo_match(self, logger):
        if False:
            print('Hello World!')
        stacktraces = ['sentry/web/urls.py']
        code_mappings = self.code_mapping_helper.generate_code_mappings(stacktraces)
        assert code_mappings == []
        logger.warning.assert_called_with('More than one repo matched sentry/web/urls.py')

    def test_list_file_matches_single(self):
        if False:
            while True:
                i = 10
        frame_filename = FrameFilename('sentry_plugins/slack/client.py')
        matches = self.code_mapping_helper.list_file_matches(frame_filename)
        expected_matches = [{'filename': 'src/sentry_plugins/slack/client.py', 'repo_name': 'Test-Organization/foo', 'repo_branch': 'master', 'stacktrace_root': 'sentry_plugins/', 'source_path': 'src/sentry_plugins/'}]
        assert matches == expected_matches

    def test_list_file_matches_multiple(self):
        if False:
            for i in range(10):
                print('nop')
        frame_filename = FrameFilename('sentry/web/urls.py')
        matches = self.code_mapping_helper.list_file_matches(frame_filename)
        expected_matches = [{'filename': 'src/sentry/web/urls.py', 'repo_name': 'Test-Organization/foo', 'repo_branch': 'master', 'stacktrace_root': 'sentry/', 'source_path': 'src/sentry/'}, {'filename': 'sentry/web/urls.py', 'repo_name': 'Test-Organization/bar', 'repo_branch': 'main', 'stacktrace_root': 'sentry/', 'source_path': 'sentry/'}]
        assert matches == expected_matches

    def test_normalized_stack_and_source_roots_starts_with_period_slash(self):
        if False:
            while True:
                i = 10
        (stacktrace_root, source_path) = self.code_mapping_helper._normalized_stack_and_source_roots('./app/', 'static/app/')
        assert stacktrace_root == './'
        assert source_path == 'static/'

    def test_normalized_stack_and_source_roots_starts_with_period_slash_no_containing_directory(self):
        if False:
            for i in range(10):
                print('nop')
        (stacktrace_root, source_path) = self.code_mapping_helper._normalized_stack_and_source_roots('./app/', 'app/')
        assert stacktrace_root == './'
        assert source_path == ''

    def test_normalized_stack_and_source_not_matching(self):
        if False:
            for i in range(10):
                print('nop')
        (stacktrace_root, source_path) = self.code_mapping_helper._normalized_stack_and_source_roots('sentry/', 'src/sentry/')
        assert stacktrace_root == 'sentry/'
        assert source_path == 'src/sentry/'

    def test_normalized_stack_and_source_roots_equal(self):
        if False:
            print('Hello World!')
        (stacktrace_root, source_path) = self.code_mapping_helper._normalized_stack_and_source_roots('source/', 'source/')
        assert stacktrace_root == ''
        assert source_path == ''

    def test_normalized_stack_and_source_roots_starts_with_period_slash_two_levels(self):
        if False:
            print('Hello World!')
        (stacktrace_root, source_path) = self.code_mapping_helper._normalized_stack_and_source_roots('./app/', 'app/foo/app/')
        assert stacktrace_root == './'
        assert source_path == 'app/foo/'

    def test_normalized_stack_and_source_roots_starts_with_app(self):
        if False:
            for i in range(10):
                print('nop')
        (stacktrace_root, source_path) = self.code_mapping_helper._normalized_stack_and_source_roots('app:///utils/', 'utils/')
        assert stacktrace_root == 'app:///'
        assert source_path == ''

    def test_normalized_stack_and_source_roots_starts_with_multiple_dot_dot_slash(self):
        if False:
            for i in range(10):
                print('nop')
        (stacktrace_root, source_path) = self.code_mapping_helper._normalized_stack_and_source_roots('../../../../../../packages/', 'packages/')
        assert stacktrace_root == '../../../../../../'
        assert source_path == ''

    def test_normalized_stack_and_source_roots_starts_with_app_dot_dot_slash(self):
        if False:
            while True:
                i = 10
        (stacktrace_root, source_path) = self.code_mapping_helper._normalized_stack_and_source_roots('app:///../services/', 'services/')
        assert stacktrace_root == 'app:///../'
        assert source_path == ''

class TestGetSortedCodeMappingConfigs(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super()
        with assume_test_silo_mode(SiloMode.CONTROL):
            self.integration = Integration.objects.create(provider='example', name='Example')
            self.integration.add_organization(self.organization, self.user)
            self.oi = OrganizationIntegration.objects.get(integration_id=self.integration.id)
        self.repo = self.create_repo(project=self.project, name='getsentry/sentry')
        self.repo.integration_id = self.integration.id
        self.repo.provider = 'example'
        self.repo.save()

    def test_get_sorted_code_mapping_configs(self):
        if False:
            print('Hello World!')
        code_mapping1 = self.create_code_mapping(organization_integration=self.oi, project=self.project, repo=self.repo, stack_root='', source_root='', automatically_generated=False)
        code_mapping2 = self.create_code_mapping(organization_integration=self.oi, project=self.project, repo=self.repo, stack_root='usr/src/getsentry/src/', source_root='', automatically_generated=True)
        code_mapping3 = self.create_code_mapping(organization_integration=self.oi, project=self.project, repo=self.repo, stack_root='usr/src/getsentry/', source_root='', automatically_generated=False)
        code_mapping4 = self.create_code_mapping(organization_integration=self.oi, project=self.project, repo=self.repo, stack_root='usr/src/', source_root='', automatically_generated=False)
        code_mapping5 = self.create_code_mapping(organization_integration=self.oi, project=self.project, repo=self.repo, stack_root='usr/src/getsentry/src/sentry/', source_root='', automatically_generated=True)
        expected_config_order = [code_mapping3, code_mapping4, code_mapping1, code_mapping5, code_mapping2]
        sorted_configs = get_sorted_code_mapping_configs(self.project)
        assert sorted_configs == expected_config_order