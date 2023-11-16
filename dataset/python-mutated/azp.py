"""Support code for working with Azure Pipelines."""
from __future__ import annotations
import os
import tempfile
import uuid
import typing as t
import urllib.parse
from ..encoding import to_bytes
from ..config import CommonConfig, TestConfig
from ..git import Git
from ..http import HttpClient
from ..util import display, MissingEnvironmentVariable
from . import ChangeDetectionNotSupported, CIProvider, CryptographyAuthHelper
CODE = 'azp'

class AzurePipelines(CIProvider):
    """CI provider implementation for Azure Pipelines."""

    def __init__(self) -> None:
        if False:
            return 10
        self.auth = AzurePipelinesAuthHelper()
        self._changes: AzurePipelinesChanges | None = None

    @staticmethod
    def is_supported() -> bool:
        if False:
            while True:
                i = 10
        'Return True if this provider is supported in the current running environment.'
        return os.environ.get('SYSTEM_COLLECTIONURI', '').startswith('https://dev.azure.com/')

    @property
    def code(self) -> str:
        if False:
            while True:
                i = 10
        'Return a unique code representing this provider.'
        return CODE

    @property
    def name(self) -> str:
        if False:
            print('Hello World!')
        'Return descriptive name for this provider.'
        return 'Azure Pipelines'

    def generate_resource_prefix(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return a resource prefix specific to this CI provider.'
        try:
            prefix = 'azp-%s-%s-%s' % (os.environ['BUILD_BUILDID'], os.environ['SYSTEM_JOBATTEMPT'], os.environ['SYSTEM_JOBIDENTIFIER'])
        except KeyError as ex:
            raise MissingEnvironmentVariable(name=ex.args[0]) from None
        return prefix

    def get_base_commit(self, args: CommonConfig) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the base commit or an empty string.'
        return self._get_changes(args).base_commit or ''

    def _get_changes(self, args: CommonConfig) -> AzurePipelinesChanges:
        if False:
            i = 10
            return i + 15
        'Return an AzurePipelinesChanges instance, which will be created on first use.'
        if not self._changes:
            self._changes = AzurePipelinesChanges(args)
        return self._changes

    def detect_changes(self, args: TestConfig) -> t.Optional[list[str]]:
        if False:
            return 10
        'Initialize change detection.'
        result = self._get_changes(args)
        if result.is_pr:
            job_type = 'pull request'
        else:
            job_type = 'merge commit'
        display.info('Processing %s for branch %s commit %s' % (job_type, result.branch, result.commit))
        if not args.metadata.changes:
            args.metadata.populate_changes(result.diff)
        if result.paths is None:
            display.warning('No successful commit found. All tests will be executed.')
        return result.paths

    def supports_core_ci_auth(self) -> bool:
        if False:
            while True:
                i = 10
        'Return True if Ansible Core CI is supported.'
        return True

    def prepare_core_ci_auth(self) -> dict[str, t.Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return authentication details for Ansible Core CI.'
        try:
            request = dict(org_name=os.environ['SYSTEM_COLLECTIONURI'].strip('/').split('/')[-1], project_name=os.environ['SYSTEM_TEAMPROJECT'], build_id=int(os.environ['BUILD_BUILDID']), task_id=str(uuid.UUID(os.environ['SYSTEM_TASKINSTANCEID'])))
        except KeyError as ex:
            raise MissingEnvironmentVariable(name=ex.args[0]) from None
        self.auth.sign_request(request)
        auth = dict(azp=request)
        return auth

    def get_git_details(self, args: CommonConfig) -> t.Optional[dict[str, t.Any]]:
        if False:
            return 10
        'Return details about git in the current environment.'
        changes = self._get_changes(args)
        details = dict(base_commit=changes.base_commit, commit=changes.commit)
        return details

class AzurePipelinesAuthHelper(CryptographyAuthHelper):
    """
    Authentication helper for Azure Pipelines.
    Based on cryptography since it is provided by the default Azure Pipelines environment.
    """

    def publish_public_key(self, public_key_pem: str) -> None:
        if False:
            i = 10
            return i + 15
        'Publish the given public key.'
        try:
            agent_temp_directory = os.environ['AGENT_TEMPDIRECTORY']
        except KeyError as ex:
            raise MissingEnvironmentVariable(name=ex.args[0]) from None
        with tempfile.NamedTemporaryFile(prefix='public-key-', suffix='.pem', delete=False, dir=agent_temp_directory) as public_key_file:
            public_key_file.write(to_bytes(public_key_pem))
            public_key_file.flush()
        vso_add_attachment('ansible-core-ci', 'public-key.pem', public_key_file.name)

class AzurePipelinesChanges:
    """Change information for an Azure Pipelines build."""

    def __init__(self, args: CommonConfig) -> None:
        if False:
            i = 10
            return i + 15
        self.args = args
        self.git = Git()
        try:
            self.org_uri = os.environ['SYSTEM_COLLECTIONURI']
            self.project = os.environ['SYSTEM_TEAMPROJECT']
            self.repo_type = os.environ['BUILD_REPOSITORY_PROVIDER']
            self.source_branch = os.environ['BUILD_SOURCEBRANCH']
            self.source_branch_name = os.environ['BUILD_SOURCEBRANCHNAME']
            self.pr_branch_name = os.environ.get('SYSTEM_PULLREQUEST_TARGETBRANCH')
        except KeyError as ex:
            raise MissingEnvironmentVariable(name=ex.args[0]) from None
        if self.source_branch.startswith('refs/tags/'):
            raise ChangeDetectionNotSupported('Change detection is not supported for tags.')
        self.org = self.org_uri.strip('/').split('/')[-1]
        self.is_pr = self.pr_branch_name is not None
        if self.is_pr:
            self.branch = self.pr_branch_name
            self.base_commit = 'HEAD^1'
            self.commit = 'HEAD^2'
        else:
            commits = self.get_successful_merge_run_commits()
            self.branch = self.source_branch_name
            self.base_commit = self.get_last_successful_commit(commits)
            self.commit = 'HEAD'
        self.commit = self.git.run_git(['rev-parse', self.commit]).strip()
        if self.base_commit:
            self.base_commit = self.git.run_git(['rev-parse', self.base_commit]).strip()
            dot_range = '%s...%s' % (self.base_commit, self.commit)
            self.paths = sorted(self.git.get_diff_names([dot_range]))
            self.diff = self.git.get_diff([dot_range])
        else:
            self.paths = None
            self.diff = []

    def get_successful_merge_run_commits(self) -> set[str]:
        if False:
            while True:
                i = 10
        'Return a set of recent successsful merge commits from Azure Pipelines.'
        parameters = dict(maxBuildsPerDefinition=100, queryOrder='queueTimeDescending', resultFilter='succeeded', reasonFilter='batchedCI', repositoryType=self.repo_type, repositoryId='%s/%s' % (self.org, self.project))
        url = '%s%s/_apis/build/builds?api-version=6.0&%s' % (self.org_uri, self.project, urllib.parse.urlencode(parameters))
        http = HttpClient(self.args, always=True)
        response = http.get(url)
        try:
            result = response.json()
        except Exception:
            display.warning('Unable to find project. Cannot determine changes. All tests will be executed.')
            return set()
        commits = set((build['sourceVersion'] for build in result['value']))
        return commits

    def get_last_successful_commit(self, commits: set[str]) -> t.Optional[str]:
        if False:
            return 10
        'Return the last successful commit from git history that is found in the given commit list, or None.'
        commit_history = self.git.get_rev_list(max_count=100)
        ordered_successful_commits = [commit for commit in commit_history if commit in commits]
        last_successful_commit = ordered_successful_commits[0] if ordered_successful_commits else None
        return last_successful_commit

def vso_add_attachment(file_type: str, file_name: str, path: str) -> None:
    if False:
        return 10
    'Upload and attach a file to the current timeline record.'
    vso('task.addattachment', dict(type=file_type, name=file_name), path)

def vso(name: str, data: dict[str, str], message: str) -> None:
    if False:
        return 10
    '\n    Write a logging command for the Azure Pipelines agent to process.\n    See: https://docs.microsoft.com/en-us/azure/devops/pipelines/scripts/logging-commands?view=azure-devops&tabs=bash\n    '
    display.info('##vso[%s %s]%s' % (name, ';'.join(('='.join((key, value)) for (key, value) in data.items())), message))