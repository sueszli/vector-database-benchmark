import gzip
import json
import os
import warnings
from hashlib import sha256
from typing import Any, Dict, List, Optional
from unittest import main, mock, skip, TestCase
from urllib.error import HTTPError
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from trymerge import categorize_checks, DRCI_CHECKRUN_NAME, find_matching_merge_rule, get_classifications, get_drci_classifications, get_rockset_results, gh_get_team_members, gh_graphql, GitHubPR, JobCheckState, main as trymerge_main, MandatoryChecksMissingError, MergeRule, PostCommentError, RE_GHSTACK_DESC, read_merge_rules, remove_job_name_suffix, validate_revert
if 'GIT_REMOTE_URL' not in os.environ:
    os.environ['GIT_REMOTE_URL'] = 'https://github.com/pytorch/pytorch'
GQL_MOCKS = 'gql_mocks.json.gz'
ROCKSET_MOCKS = 'rockset_mocks.json.gz'
DRCI_MOCKS = 'drci_mocks.json.gz'

def mock_query(fallback_function: Any, file_name: str, key_function: Any, *args: Any) -> Any:
    if False:
        while True:
            i = 10
    gql_db_fname = os.path.join(os.path.dirname(__file__), file_name)

    def get_mocked_queries() -> Any:
        if False:
            i = 10
            return i + 15
        if not os.path.exists(gql_db_fname):
            return {}
        with gzip.open(gql_db_fname, encoding='utf-8', mode='rt') as f:
            return json.load(f)

    def save_mocked_queries(obj: Any) -> None:
        if False:
            return 10
        with gzip.open(gql_db_fname, encoding='utf-8', mode='wt') as f:
            json.dump(obj, f, indent=2)
            f.write('\n')
    key = key_function(*args)
    mocked_queries = get_mocked_queries()
    if key in mocked_queries:
        return mocked_queries[key]
    try:
        rc = fallback_function(*args)
    except HTTPError as err:
        if err.code == 401 or err.code == 403:
            err_msg = f'If you are seeing this message during workflow run, please make sure to update {file_name}'
            err_msg += f' locally, by deleting it and running {os.path.basename(__file__)} with'
            err_msg += ' GitHub Personal Access Token passed via GITHUB_TOKEN,'
            err_msg += ' the rockset api key passed via ROCKSET_API_KEY,'
            err_msg += ' and drci api key passed via DRCI_BOT_KEY environment variables'
            if os.getenv('GITHUB_TOKEN') is None or os.getenv('ROCKSET_API_KEY') is None or os.getenv('DRCI_BOT_KEY') is None:
                err_msg = 'Failed to update cached queries as GITHUB_TOKEN or ROCKSET_API_KEY or DRCI_BOT_KEY ' + 'is not defined. ' + err_msg
            raise RuntimeError(err_msg) from err
    mocked_queries[key] = rc
    save_mocked_queries(mocked_queries)
    return rc

def mocked_gh_graphql(query: str, **kwargs: Any) -> Any:
    if False:
        return 10

    def key_function(query: str, kwargs: Any) -> str:
        if False:
            while True:
                i = 10
        return f"query_sha={sha256(query.encode('utf-8')).hexdigest()} " + ' '.join([f'{k}={kwargs[k]}' for k in sorted(kwargs.keys())])

    def gh_graphql_wrapper(query: str, kwargs: Any) -> Any:
        if False:
            while True:
                i = 10
        return gh_graphql(query, **kwargs)
    return mock_query(gh_graphql_wrapper, GQL_MOCKS, key_function, query, kwargs)

def mocked_rockset_results(head_sha: str, merge_base: str, num_retries: int=3) -> Any:
    if False:
        for i in range(10):
            print('nop')
    return mock_query(get_rockset_results, ROCKSET_MOCKS, lambda x, y: f'{x} {y}', head_sha, merge_base)

def mocked_drci_classifications(pr_num: int, project: str, num_retries: int=3) -> Any:
    if False:
        print('Hello World!')
    return mock_query(get_drci_classifications, DRCI_MOCKS, lambda x, y: f'{x} {y}', pr_num, project)

def mock_parse_args(revert: bool=False, force: bool=False) -> Any:
    if False:
        print('Hello World!')

    class Object:

        def __init__(self) -> None:
            if False:
                print('Hello World!')
            self.revert = revert
            self.force = force
            self.pr_num = 76123
            self.dry_run = True
            self.comment_id = 0
            self.reason = 'this is for testing'
            self.ignore_current = False
    return Object()

def mock_remove_label(org: str, repo: str, pr_num: str, label: str) -> None:
    if False:
        print('Hello World!')
    pass

def mock_revert(repo: GitRepo, pr: GitHubPR, *, dry_run: bool=False, comment_id: Optional[int]=None, reason: Optional[str]=None) -> None:
    if False:
        while True:
            i = 10
    pass

def mock_merge(pr: GitHubPR, repo: GitRepo, dry_run: bool=False, skip_mandatory_checks: bool=False, comment_id: Optional[int]=None, timeout_minutes: int=400, stale_pr_days: int=3, ignore_current: bool=False) -> None:
    if False:
        return 10
    pass

def mock_gh_get_info() -> Any:
    if False:
        return 10
    return {'closed': False, 'isCrossRepository': False, 'files': {'nodes': [], 'pageInfo': {'hasNextPage': False}}, 'changedFiles': 0}

def mocked_read_merge_rules_NE(repo: Any, org: str, project: str) -> List[MergeRule]:
    if False:
        while True:
            i = 10
    return [MergeRule(name='mock with nonexistent check', patterns=['*'], approved_by=[], mandatory_checks_name=['Lint', 'Facebook CLA Check', 'nonexistent'], ignore_flaky_failures=True)]

def mocked_read_merge_rules(repo: Any, org: str, project: str) -> List[MergeRule]:
    if False:
        return 10
    return [MergeRule(name='super', patterns=['*'], approved_by=['pytorch/metamates', 'ngimel'], mandatory_checks_name=['Lint', 'Facebook CLA Check', 'pull / linux-xenial-cuda11.3-py3.7-gcc7 / build'], ignore_flaky_failures=True), MergeRule(name='xla', patterns=['.github/ci_commit_pins/xla.txt'], approved_by=['pytorchbot'], mandatory_checks_name=['Lint', 'EasyCLA', 'pull / linux-focal-py3_8-clang9-xla / build', 'pull / linux-focal-py3_8-clang9-xla / test (xla, 1, 1, linux.12xlarge)'], ignore_flaky_failures=True)]

def mocked_read_merge_rules_raise(repo: Any, org: str, project: str) -> List[MergeRule]:
    if False:
        while True:
            i = 10
    raise RuntimeError('testing')

def xla_merge_rules(repo: Any, org: str, project: str) -> List[MergeRule]:
    if False:
        for i in range(10):
            print('nop')
    return [MergeRule(name=' OSS CI / pytorchbot / XLA', patterns=['.github/ci_commit_pins/xla.txt'], approved_by=['pytorchbot'], mandatory_checks_name=['Lint', 'EasyCLA', 'pull / linux-bionic-py3_8-clang8-xla / build', 'pull / linux-bionic-py3_8-clang8-xla / test (xla, 1, 1, linux.4xlarge)', 'inductor / cuda11.8-py3.10-gcc7-sm86 / test (inductor_torchbench_dynamic, 1, 1, linux.g5.4xlarge.nvidia.gpu)'], ignore_flaky_failures=False)]

def empty_rockset_results(head_sha: str, merge_base: str) -> List[Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    return []

class DummyGitRepo(GitRepo):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(get_git_repo_dir(), get_git_remote_name())

    def commits_resolving_gh_pr(self, pr_num: int) -> List[str]:
        if False:
            while True:
                i = 10
        return ['FakeCommitSha']

    def commit_message(self, ref: str) -> str:
        if False:
            return 10
        return 'super awsome commit message'

@mock.patch('trymerge.get_rockset_results', side_effect=empty_rockset_results)
@mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
@mock.patch('trymerge.get_drci_classifications', side_effect=mocked_drci_classifications)
class TestTryMerge(TestCase):

    def test_merge_rules_valid(self, *args: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Test that merge_rules.yaml can be parsed'
        repo = DummyGitRepo()
        merge_rules = read_merge_rules(repo, 'pytorch', 'pytorch')
        self.assertGreater(len(merge_rules), 1)

    @mock.patch('trymerge.read_merge_rules', side_effect=mocked_read_merge_rules)
    def test_match_rules(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that PR passes merge rules'
        pr = GitHubPR('pytorch', 'pytorch', 109999)
        repo = DummyGitRepo()
        self.assertTrue(find_matching_merge_rule(pr, repo) is not None)

    @mock.patch('trymerge.read_merge_rules', side_effect=mocked_read_merge_rules_raise)
    def test_read_merge_rules_fails(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that PR fails to read the merge rules'
        pr = GitHubPR('pytorch', 'pytorch', 77700)
        repo = DummyGitRepo()
        self.assertRaisesRegex(RuntimeError, 'testing', lambda : find_matching_merge_rule(pr, repo))

    @mock.patch('trymerge.read_merge_rules', side_effect=mocked_read_merge_rules)
    def test_lint_fails(self, *args: Any) -> None:
        if False:
            return 10
        'Tests that PR fails mandatory lint check'
        pr = GitHubPR('pytorch', 'pytorch', 90791)
        repo = DummyGitRepo()
        self.assertRaises(RuntimeError, lambda : find_matching_merge_rule(pr, repo))

    def test_get_last_comment(self, *args: Any) -> None:
        if False:
            while True:
                i = 10
        'Tests that last comment can be fetched'
        pr = GitHubPR('pytorch', 'pytorch', 71759)
        comment = pr.get_last_comment()
        self.assertEqual(comment.author_login, 'github-actions')
        self.assertIsNone(comment.editor_login)
        self.assertTrue("You've committed this PR" in comment.body_text)

    def test_get_author_null(self, *args: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that PR author can be computed\n        If reply contains NULL\n        '
        pr = GitHubPR('pytorch', 'pytorch', 71759)
        author = pr.get_author()
        self.assertTrue(author is not None)
        self.assertTrue('@' in author)
        self.assertTrue(pr.get_diff_revision() is None)
        pr = GitHubPR('pytorch', 'pytorch', 75095)
        self.assertEqual(pr.get_pr_creator_login(), 'mruberry')
        author = pr.get_author()
        self.assertTrue(author is not None)

    def test_large_diff(self, *args: Any) -> None:
        if False:
            while True:
                i = 10
        'Tests that PR with 100+ files can be fetched'
        pr = GitHubPR('pytorch', 'pytorch', 73099)
        self.assertTrue(pr.get_changed_files_count() > 100)
        flist = pr.get_changed_files()
        self.assertEqual(len(flist), pr.get_changed_files_count())

    def test_internal_changes(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that PR with internal changes is detected'
        pr = GitHubPR('pytorch', 'pytorch', 110140)
        self.assertTrue(pr.has_internal_changes())

    def test_comments_pagination(self, *args: Any) -> None:
        if False:
            print('Hello World!')
        'Tests that PR with 50+ comments can be fetched'
        pr = GitHubPR('pytorch', 'pytorch', 31093)
        self.assertGreater(len(pr.get_comments()), 50)

    def test_gql_complexity(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Fetch comments and conclusions for PR with 60 commits'
        pr = GitHubPR('pytorch', 'pytorch', 68111)
        self.assertGreater(len(pr.get_comments()), 20)
        self.assertGreater(pr.get_commit_count(), 60)

    def test_gql_retrieve_checksuites(self, *args: Any) -> None:
        if False:
            while True:
                i = 10
        'Fetch comments and conclusions for PR with 60 commits'
        pr = GitHubPR('pytorch', 'pytorch', 94787)
        self.assertEqual(len(pr.get_checkrun_conclusions()), 183)

    def test_team_members(self, *args: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Test fetching team members works'
        dev_infra_team = gh_get_team_members('pytorch', 'pytorch-dev-infra')
        self.assertGreater(len(dev_infra_team), 2)
        with self.assertWarns(Warning):
            non_existing_team = gh_get_team_members('pytorch', 'qwertyuiop')
            self.assertEqual(len(non_existing_team), 0)

    def test_get_author_many_commits(self, *args: Any) -> None:
        if False:
            while True:
                i = 10
        'Tests that authors for all commits can be fetched'
        pr = GitHubPR('pytorch', 'pytorch', 76118)
        authors = pr.get_authors()
        self.assertGreater(pr.get_commit_count(), 100)
        self.assertGreater(len(authors), 50)
        self.assertTrue('@' in pr.get_author())

    @mock.patch('trymerge.read_merge_rules', side_effect=mocked_read_merge_rules_NE)
    def test_pending_status_check(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that PR with nonexistent/pending status checks fails with the right reason.'
        pr = GitHubPR('pytorch', 'pytorch', 76118)
        repo = DummyGitRepo()
        self.assertRaisesRegex(MandatoryChecksMissingError, '.*are pending/not yet run.*', lambda : find_matching_merge_rule(pr, repo))

    def test_get_author_many_reviews(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that all reviews can be fetched'
        pr = GitHubPR('pytorch', 'pytorch', 76123)
        approved_by = pr.get_approved_by()
        self.assertGreater(len(approved_by), 0)
        assert pr._reviews is not None
        self.assertGreater(len(pr._reviews), 100)

    def test_get_checkruns_many_runs(self, *args: Any) -> None:
        if False:
            while True:
                i = 10
        'Tests that all checkruns can be fetched'
        pr = GitHubPR('pytorch', 'pytorch', 105260)
        conclusions = pr.get_checkrun_conclusions()
        self.assertEqual(len(conclusions), 221)
        self.assertTrue('pull / linux-docs / build-docs-cpp-false' in conclusions.keys())

    def test_cancelled_gets_ignored(self, *args: Any) -> None:
        if False:
            return 10
        'Tests that cancelled workflow does not override existing successfull status'
        pr = GitHubPR('pytorch', 'pytorch', 110367)
        conclusions = pr.get_checkrun_conclusions()
        lint_checks = [name for name in conclusions.keys() if 'Lint' in name]
        self.assertTrue(len(lint_checks) > 0)
        self.assertTrue(all((conclusions[name].status == 'SUCCESS' for name in lint_checks)))

    def test_get_review_comment_by_id(self, *args: Any) -> None:
        if False:
            return 10
        'Tests that even if the comment requested was actually a review instead of a simple comment, we can still find it'
        pr = GitHubPR('pytorch', 'pytorch', 107070)
        review_comment_id = 1582767635
        comment = pr.get_comment_by_id(review_comment_id)
        self.assertIsNotNone(comment)

    @mock.patch('trymerge.gh_get_pr_info', return_value=mock_gh_get_info())
    @mock.patch('trymerge.parse_args', return_value=mock_parse_args(True, False))
    @mock.patch('trymerge.try_revert', side_effect=mock_revert)
    def test_main_revert(self, mock_revert: Any, *args: Any) -> None:
        if False:
            return 10
        trymerge_main()
        mock_revert.assert_called_once()

    @mock.patch('trymerge.gh_get_pr_info', return_value=mock_gh_get_info())
    @mock.patch('trymerge.parse_args', return_value=mock_parse_args(False, True))
    @mock.patch('trymerge.gh_remove_label', side_effect=mock_remove_label)
    @mock.patch('trymerge.merge', side_effect=mock_merge)
    def test_main_force(self, mock_merge: Any, mock_parse_args: Any, *args: Any) -> None:
        if False:
            print('Hello World!')
        trymerge_main()
        mock_merge.assert_called_once_with(mock.ANY, mock.ANY, dry_run=mock.ANY, skip_mandatory_checks=True, comment_id=mock.ANY, ignore_current=False)

    @mock.patch('trymerge.gh_get_pr_info', return_value=mock_gh_get_info())
    @mock.patch('trymerge.parse_args', return_value=mock_parse_args(False, False))
    @mock.patch('trymerge.gh_remove_label', side_effect=mock_remove_label)
    @mock.patch('trymerge.merge', side_effect=mock_merge)
    def test_main_merge(self, mock_merge: Any, *args: Any) -> None:
        if False:
            return 10
        trymerge_main()
        mock_merge.assert_called_once_with(mock.ANY, mock.ANY, dry_run=mock.ANY, skip_mandatory_checks=False, comment_id=mock.ANY, ignore_current=False)

    @mock.patch('trymerge.read_merge_rules', side_effect=mocked_read_merge_rules)
    def test_revert_rules(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that reverts from collaborators are allowed'
        pr = GitHubPR('pytorch', 'pytorch', 79694)
        repo = DummyGitRepo()
        self.assertIsNotNone(validate_revert(repo, pr, comment_id=1189459845))

    def test_get_changed_files(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Tests that the list changed files in a PR doesn't include duplicates\n        "
        pr = GitHubPR('pytorch', 'pytorch', 95233)
        try:
            changed_files = pr.get_changed_files()
        except RuntimeError as error:
            self.fail(f'get_changed_files throws an exception: {error}')
        self.assertEqual(len(changed_files), pr.get_changed_files_count())

    def test_revert_codev_fails(self, *args: Any) -> None:
        if False:
            print('Hello World!')
        pr = GitHubPR('pytorch', 'pytorch', 91340)

        class GitRepoCoDev(DummyGitRepo):

            def commit_message(self, ref: str) -> str:
                if False:
                    for i in range(10):
                        print('nop')
                return pr.get_body()
        repo = GitRepoCoDev()
        self.assertRaisesRegex(PostCommentError, 'landed via phabricator', lambda : validate_revert(repo, pr, comment_id=1372496233))

    def test_revert_codev_abandoned_diff_succeeds(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        pr = GitHubPR('pytorch', 'pytorch', 100652)

        class GitRepoCoDev(DummyGitRepo):

            def commit_message(self, ref: str) -> str:
                if False:
                    print('Hello World!')
                return pr.get_body()
        repo = GitRepoCoDev()
        validate_revert(repo, pr, comment_id=1588195237)

    def test_pr_changed_submodule_detection(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        pr = GitHubPR('pytorch', 'pytorch', 95045)
        self.assertEqual(pr.get_changed_submodules(), [])
        self.assertFalse(pr.has_invalid_submodule_updates())
        pr = GitHubPR('pytorch', 'pytorch', 94939)
        self.assertEqual(pr.get_changed_submodules(), ['third_party/ideep'])
        self.assertTrue(pr.has_invalid_submodule_updates())
        pr = GitHubPR('pytorch', 'pytorch', 91051)
        self.assertEqual(pr.get_changed_submodules(), ['third_party/kineto'])
        self.assertFalse(pr.has_invalid_submodule_updates())

    def test_remove_job_name_suffix(self, *args: Any) -> None:
        if False:
            print('Hello World!')
        test_cases = [{'name': 'linux-bionic-cuda12.1-py3.10-gcc9-sm86 / test (default, 1, 5, linux.g5.4xlarge.nvidia.gpu)', 'expected': 'linux-bionic-cuda12.1-py3.10-gcc9-sm86 / test (default)'}, {'name': 'android-emulator-build-test / build-and-test (default, 1, 1, ubuntu-20.04-16x)', 'expected': 'android-emulator-build-test / build-and-test (default)'}, {'name': 'linux-focal-rocm5.4.2-py3.8 / build', 'expected': 'linux-focal-rocm5.4.2-py3.8 / build'}, {'name': 'libtorch-cpu-shared-with-deps-release-build', 'expected': 'libtorch-cpu-shared-with-deps-release-build'}, {'name': 'manywheel-py3_8-cuda11_8-test / test', 'expected': 'manywheel-py3_8-cuda11_8-test / test'}, {'name': 'lintrunner / linux-job', 'expected': 'lintrunner / linux-job'}, {'name': 'Test `run_test.py` is usable without boto3/rockset', 'expected': 'Test `run_test.py` is usable without boto3/rockset'}]
        for case in test_cases:
            self.assertEqual(case['expected'], remove_job_name_suffix(case['name']))

    def test_get_merge_base(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        pr = GitHubPR('pytorch', 'pytorch', 104121)
        mock_merge_base = 'mocked-sha'
        with mock.patch('trymerge.gh_fetch_merge_base', return_value=mock_merge_base) as mocked_gh_fetch_merge_base:
            self.assertEqual(mock_merge_base, pr.get_merge_base())
            self.assertEqual(mock_merge_base, pr.get_merge_base())
            mocked_gh_fetch_merge_base.assert_called_once()

@mock.patch('trymerge.get_rockset_results', side_effect=mocked_rockset_results)
@mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
@mock.patch('trymerge.gh_fetch_merge_base', return_value='')
@mock.patch('trymerge.get_drci_classifications', side_effect=mocked_drci_classifications)
class TestBypassFailures(TestCase):

    def test_get_classifications(self, *args: Any) -> None:
        if False:
            print('Hello World!')
        pr = GitHubPR('pytorch', 'pytorch', 109584)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(pr.pr_num, pr.project, checks, [])
        self.assertTrue(checks['pull / linux-focal-py3.11-clang10 / test (dynamo, 1, 2, linux.2xlarge)'].classification == 'BROKEN_TRUNK')
        self.assertTrue(checks['trunk / win-vs2019-cpu-py3 / test (default, 2, 3, windows.4xlarge.nonephemeral)'].classification == 'FLAKY')
        self.assertTrue(checks['pull / linux-jammy-py3.8-gcc11 / test (distributed, 1, 2, linux.2xlarge)'].classification == 'FLAKY')
        self.assertTrue(checks['pull / linux-focal-cuda11.8-py3.10-gcc9 / test (distributed, 1, 3, linux.8xlarge.nvidia.gpu)'].classification == 'FLAKY')
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()), ok_failed_checks_threshold=6)
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable['FLAKY']) == 4)
        self.assertTrue(len(ignorable['BROKEN_TRUNK']) == 2)
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable['FLAKY']) == 4)
        self.assertTrue(len(ignorable['BROKEN_TRUNK']) == 2)
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()), ok_failed_checks_threshold=1)
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 6)
        self.assertTrue(len(ignorable['FLAKY']) == 4)
        self.assertTrue(len(ignorable['BROKEN_TRUNK']) == 2)
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()), ok_failed_checks_threshold=1)
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 6)
        self.assertTrue(len(ignorable['FLAKY']) == 4)
        self.assertTrue(len(ignorable['BROKEN_TRUNK']) == 2)

    def test_get_classifications_flaky_fullname(self, *args: Any) -> None:
        if False:
            print('Hello World!')
        pr = GitHubPR('pytorch', 'pytorch', 110362)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(pr.pr_num, pr.project, checks, [])
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable['FLAKY']) == 1)

    def test_get_classifications_invalid_cancel(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        pr = GitHubPR('pytorch', 'pytorch', 110367)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(pr.pr_num, pr.project, checks, [])
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable['FLAKY']) == 0)
        self.assertTrue(len(ignorable['BROKEN_TRUNK']) == 0)
        self.assertTrue(len(ignorable['UNSTABLE']) == 3)

    def test_get_classifications_similar_failures(self, *args: Any) -> None:
        if False:
            print('Hello World!')
        pr = GitHubPR('pytorch', 'pytorch', 109750)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(pr.pr_num, pr.project, checks, [])
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable['FLAKY']) == 1)

    def test_get_classifications_unstable(self, *args: Any) -> None:
        if False:
            while True:
                i = 10
        pr = GitHubPR('pytorch', 'pytorch', 104312)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(pr.pr_num, pr.project, checks, [])
        workflow_name = 'linux-bionic-cuda12.1-py3.10-gcc9-bazel-test'
        job_name = 'build-and-test (default, 1, 1, linux.4xlarge.nvidia.gpu, unstable)'
        self.assertTrue(checks[f'pull / {workflow_name} / {job_name}'].classification == 'UNSTABLE')
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()), ok_failed_checks_threshold=1)
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable['UNSTABLE']) == 1)

    def test_get_classifications_broken_trunk(self, *args: Any) -> None:
        if False:
            return 10
        test_cases = [{'pr_num': 104214, 'related_failure_count': 0, 'unrelated_failure_count': 1}, {'pr_num': 105145, 'related_failure_count': 0, 'unrelated_failure_count': 1}, {'pr_num': 107160, 'related_failure_count': 0, 'unrelated_failure_count': 4}, {'pr_num': 111253, 'related_failure_count': 1, 'unrelated_failure_count': 2}]
        for case in test_cases:
            pr_num = case['pr_num']
            related_failure_count = case['related_failure_count']
            unrelated_failure_count = case['unrelated_failure_count']
            pr = GitHubPR('pytorch', 'pytorch', pr_num)
            checks = pr.get_checkrun_conclusions()
            checks = get_classifications(pr.pr_num, pr.project, checks, [])
            (pending, failed, _) = categorize_checks(checks, list(checks.keys()))
            self.assertTrue(len(pending) == 0)
            self.assertTrue(len(failed) == related_failure_count)
            (pending, failed, _) = categorize_checks(checks, list(checks.keys()), ok_failed_checks_threshold=0)
            self.assertTrue(len(pending) == 0)
            self.assertTrue(len(failed) == unrelated_failure_count + related_failure_count)

    def test_ignore_current(self, *args: Any) -> None:
        if False:
            i = 10
            return i + 15
        flaky = 'pull / linux-focal-cuda11.8-py3.10-gcc9 / test (distributed, 1, 3, linux.8xlarge.nvidia.gpu)'
        broken_trunk = 'pull / linux-focal-py3.11-clang10 / test (dynamo, 1, 2, linux.2xlarge)'
        pr = GitHubPR('pytorch', 'pytorch', 109584)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(pr.pr_num, pr.project, checks, [broken_trunk, flaky])
        self.assertTrue(checks[flaky].classification == 'FLAKY')
        self.assertTrue(checks[broken_trunk].classification == 'BROKEN_TRUNK')
        (_, failed, ignorable) = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable['IGNORE_CURRENT_CHECK']) == 0)
        self.assertTrue(len(ignorable['FLAKY']) == 4)
        self.assertTrue(len(ignorable['BROKEN_TRUNK']) == 2)

    @mock.patch('trymerge.read_merge_rules', side_effect=xla_merge_rules)
    def test_dont_ignore_flaky_failures(self, *args: Any) -> None:
        if False:
            return 10
        '\n        Regression test for https://github.com/pytorch/test-infra/issues/4126\n        '
        pr = GitHubPR('pytorch', 'pytorch', 105312)
        repo = DummyGitRepo()
        with warnings.catch_warnings(record=True) as w, self.assertRaises(RuntimeError):
            rule = find_matching_merge_rule(pr, repo)
        self.assertEqual(len(w), 1)
        self.assertIn('1 checks failed but were likely due flakiness or broken trunk', str(w[0].message))

@mock.patch('trymerge.get_rockset_results', side_effect=mocked_rockset_results)
@mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
@mock.patch('trymerge.gh_fetch_merge_base', return_value='')
@mock.patch('trymerge.get_drci_classifications', return_value={})
class TestBypassFailuresOnSandCastle(TestCase):

    def test_get_classifications(self, *args: Any) -> None:
        if False:
            i = 10
            return i + 15
        pr = GitHubPR('pytorch', 'pytorch', 111467)
        checks = pr.get_checkrun_conclusions()
        checks = get_classifications(pr.pr_num, pr.project, checks, [])
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 0)
        self.assertTrue(len(ignorable['FLAKY']) == 1)
        self.assertTrue(len(ignorable['BROKEN_TRUNK']) == 1)

    def test_get_classifications_drci_checkrun_not_found(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        pr = GitHubPR('pytorch', 'pytorch', 111467)
        checks = pr.get_checkrun_conclusions()
        checks[DRCI_CHECKRUN_NAME] = JobCheckState(DRCI_CHECKRUN_NAME, '', 'NEUTRAL', None, 1, '', None)
        checks = get_classifications(pr.pr_num, pr.project, checks, [])
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 2)
        checks = pr.get_checkrun_conclusions()
        checks[DRCI_CHECKRUN_NAME] = JobCheckState(DRCI_CHECKRUN_NAME, '', 'NEUTRAL', None, 1, '', '')
        checks = get_classifications(pr.pr_num, pr.project, checks, [])
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 2)
        checks = pr.get_checkrun_conclusions()
        del checks[DRCI_CHECKRUN_NAME]
        checks = get_classifications(pr.pr_num, pr.project, checks, [])
        (pending, failed, ignorable) = categorize_checks(checks, list(checks.keys()))
        self.assertTrue(len(pending) == 0)
        self.assertTrue(len(failed) == 2)

@mock.patch('trymerge.get_rockset_results', side_effect=mocked_rockset_results)
@mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
@mock.patch('trymerge.gh_fetch_merge_base', return_value='')
@mock.patch('trymerge.get_drci_classifications', side_effect=mocked_drci_classifications)
class TestGitHubPRGhstackDependencies(TestCase):

    def test_pr_dependencies(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        pr = GitHubPR('pytorch', 'pytorch', 106068)
        msg = pr.gen_commit_message(filter_ghstack=True)
        self.assertEqual(msg, f"{pr.get_title()} (#106068)\n\n{RE_GHSTACK_DESC.sub('', pr.get_body())}\nPull Request resolved: https://github.com/pytorch/pytorch/pull/106068\nApproved by: https://github.com/ezyang, https://github.com/fegin\n")

    def test_pr_dependencies_ghstack(self, *args: Any) -> None:
        if False:
            return 10
        pr0 = GitHubPR('pytorch', 'pytorch', 106032)
        pr1 = GitHubPR('pytorch', 'pytorch', 106033)
        pr2 = GitHubPR('pytorch', 'pytorch', 106034)
        pr = GitHubPR('pytorch', 'pytorch', 106068)
        msg = pr.gen_commit_message(filter_ghstack=True, ghstack_deps=[pr0, pr1, pr2])
        self.assertEqual(msg, f"{pr.get_title()} (#106068)\n\n{RE_GHSTACK_DESC.sub('', pr.get_body())}\nPull Request resolved: https://github.com/pytorch/pytorch/pull/106068\nApproved by: https://github.com/ezyang, https://github.com/fegin\nghstack dependencies: #106032, #106033, #106034\n")

    @skip(reason='This test is run against a mutalbe PR that has changed, so it no longer works. The test should be changed')
    @mock.patch('trymerge.read_merge_rules')
    @mock.patch('trymerge.GitRepo')
    @mock.patch('trymerge.get_ghstack_prs')
    def test_merge_ghstack_into(self, mock_get_ghstack_prs: mock.MagicMock, mock_repo: mock.MagicMock, mock_merge_rules: mock.MagicMock, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that the merge_ghstack_into method works correctly\n        '
        pr0 = GitHubPR('pytorch', 'pytorch', 106032)
        pr1 = GitHubPR('pytorch', 'pytorch', 106033)
        pr2 = GitHubPR('pytorch', 'pytorch', 106034)
        pr = GitHubPR('pytorch', 'pytorch', 106068)
        mock_get_ghstack_prs.return_value = [(pr0, 'rev0'), (pr1, 'rev1'), (pr2, 'rev2'), (pr, 'rev123')]
        mock_merge_rules.return_value = [MergeRule('Mock title', patterns=['*'], approved_by=[], mandatory_checks_name=None)]
        mock_repo.cherry_pick.return_value = None
        mock_repo.amend_commit_message.return_value = None
        res = pr.merge_ghstack_into(mock_repo, True)
        self.assertEqual(res, [pr2, pr])
        mock_repo.cherry_pick.assert_any_call('rev2')
        mock_repo.cherry_pick.assert_any_call('rev123')
        self.assertTrue(mock.call('rev1') not in mock_repo.cherry_pick.call_args_list)
        message = mock_repo.amend_commit_message.call_args_list[0].args[0]
        prefix = '[FSDP] Optimize away intermediate `div_` for HSDP (#106034)\n\n\r\n### Background: Gradient Pre-Divide'
        suffix = '\nPull Request resolved: https://github.com/pytorch/pytorch/pull/106034\nApproved by: \nghstack dependencies: #106032, #106033\n'
        self.assertTrue(message.startswith(prefix))
        self.assertTrue(message.endswith(suffix))
        mock_repo.amend_commit_message.assert_any_call('[FSDP] Break up `_post_backward_hook` into smaller funcs (#106068)\n\n\nDifferential Revision: [D47852461](https://our.internmc.facebook.com/intern/diff/D47852461)\nPull Request resolved: https://github.com/pytorch/pytorch/pull/106068\nApproved by: \nghstack dependencies: #106032, #106033, #106034\n')
if __name__ == '__main__':
    main()