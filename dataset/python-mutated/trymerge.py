import base64
import json
import os
import re
import time
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, cast, Dict, List, NamedTuple, Optional, Pattern, Tuple
from warnings import warn
import yaml
from github_utils import gh_fetch_json_list, gh_fetch_merge_base, gh_fetch_url, gh_post_commit_comment, gh_post_pr_comment, gh_update_pr_state, GitHubComment
from gitutils import are_ghstack_branches_in_sync, get_git_remote_name, get_git_repo_dir, GitRepo, patterns_to_regex, retries_decorator
from label_utils import gh_add_labels, gh_remove_label, has_required_labels, LABEL_ERR_MSG
from trymerge_explainer import get_revert_message, TryMergeExplainer
MERGE_IN_PROGRESS_LABEL = 'merging'
MERGE_COMPLETE_LABEL = 'merged'

class JobCheckState(NamedTuple):
    name: str
    url: str
    status: Optional[str]
    classification: Optional[str]
    job_id: Optional[int]
    title: Optional[str]
    summary: Optional[str]
JobNameToStateDict = Dict[str, JobCheckState]

class WorkflowCheckState:

    def __init__(self, name: str, url: str, status: Optional[str]):
        if False:
            print('Hello World!')
        self.name: str = name
        self.url: str = url
        self.status: Optional[str] = status
        self.jobs: JobNameToStateDict = {}
GH_PR_REVIEWS_FRAGMENT = '\nfragment PRReviews on PullRequestReviewConnection {\n  nodes {\n    author {\n      login\n    }\n    bodyText\n    createdAt\n    authorAssociation\n    editor {\n      login\n    }\n    databaseId\n    url\n    state\n  }\n  pageInfo {\n    startCursor\n    hasPreviousPage\n  }\n}\n'
GH_CHECKSUITES_FRAGMENT = '\nfragment PRCheckSuites on CheckSuiteConnection {\n  edges {\n    node {\n      app {\n        name\n        databaseId\n      }\n      workflowRun {\n        workflow {\n          name\n        }\n        url\n      }\n      checkRuns(first: 50) {\n        nodes {\n          name\n          conclusion\n          detailsUrl\n          databaseId\n          title\n          summary\n        }\n        pageInfo {\n          endCursor\n          hasNextPage\n        }\n      }\n      conclusion\n    }\n    cursor\n  }\n  pageInfo {\n    hasNextPage\n  }\n}\n'
GH_COMMIT_AUTHORS_FRAGMENT = '\nfragment CommitAuthors on PullRequestCommitConnection {\n  nodes {\n    commit {\n      author {\n        user {\n          login\n        }\n        email\n        name\n      }\n      oid\n    }\n  }\n  pageInfo {\n    endCursor\n    hasNextPage\n  }\n}\n'
GH_GET_PR_INFO_QUERY = GH_PR_REVIEWS_FRAGMENT + GH_CHECKSUITES_FRAGMENT + GH_COMMIT_AUTHORS_FRAGMENT + '\nquery ($owner: String!, $name: String!, $number: Int!) {\n  repository(owner: $owner, name: $name) {\n    pullRequest(number: $number) {\n      closed\n      isCrossRepository\n      author {\n        login\n      }\n      title\n      body\n      headRefName\n      headRepository {\n        nameWithOwner\n      }\n      baseRefName\n      baseRefOid\n      baseRepository {\n        nameWithOwner\n        isPrivate\n        defaultBranchRef {\n          name\n        }\n      }\n      mergeCommit {\n        oid\n      }\n      commits_with_authors: commits(first: 100) {\n        ...CommitAuthors\n        totalCount\n      }\n      commits(last: 1) {\n        nodes {\n          commit {\n            checkSuites(first: 10) {\n              ...PRCheckSuites\n            }\n            status {\n              contexts {\n                context\n                state\n                targetUrl\n              }\n            }\n            oid\n          }\n        }\n      }\n      changedFiles\n      files(first: 100) {\n        nodes {\n          path\n        }\n        pageInfo {\n          endCursor\n          hasNextPage\n        }\n      }\n      reviews(last: 100) {\n        ...PRReviews\n      }\n      comments(last: 5) {\n        nodes {\n          bodyText\n          createdAt\n          author {\n            login\n          }\n          authorAssociation\n          editor {\n            login\n          }\n          databaseId\n          url\n        }\n        pageInfo {\n          startCursor\n          hasPreviousPage\n        }\n      }\n      labels(first: 100) {\n        edges {\n          node {\n            name\n          }\n        }\n      }\n    }\n  }\n}\n'
GH_GET_PR_NEXT_FILES_QUERY = '\nquery ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {\n  repository(name: $name, owner: $owner) {\n    pullRequest(number: $number) {\n      files(first: 100, after: $cursor) {\n        nodes {\n          path\n        }\n        pageInfo {\n          endCursor\n          hasNextPage\n        }\n      }\n    }\n  }\n}\n'
GH_GET_PR_NEXT_CHECKSUITES = GH_CHECKSUITES_FRAGMENT + '\nquery ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {\n  repository(name: $name, owner: $owner) {\n    pullRequest(number: $number) {\n      commits(last: 1) {\n        nodes {\n          commit {\n            oid\n            checkSuites(first: 10, after: $cursor) {\n              ...PRCheckSuites\n            }\n          }\n        }\n      }\n    }\n  }\n}\n'
GH_GET_PR_NEXT_CHECK_RUNS = '\nquery ($owner: String!, $name: String!, $number: Int!, $cs_cursor: String, $cr_cursor: String!) {\n  repository(name: $name, owner: $owner) {\n    pullRequest(number: $number) {\n      commits(last: 1) {\n        nodes {\n          commit {\n            oid\n            checkSuites(first: 1, after: $cs_cursor) {\n              nodes {\n                checkRuns(first: 100, after: $cr_cursor) {\n                  nodes {\n                    name\n                    conclusion\n                    detailsUrl\n                    databaseId\n                    title\n                    summary\n                  }\n                  pageInfo {\n                    endCursor\n                    hasNextPage\n                  }\n                }\n              }\n            }\n          }\n        }\n      }\n    }\n  }\n}\n'
GH_GET_PR_PREV_COMMENTS = '\nquery ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {\n  repository(name: $name, owner: $owner) {\n    pullRequest(number: $number) {\n      comments(last: 100, before: $cursor) {\n        nodes {\n          bodyText\n          createdAt\n          author {\n            login\n          }\n          authorAssociation\n          editor {\n            login\n          }\n          databaseId\n          url\n        }\n        pageInfo {\n          startCursor\n          hasPreviousPage\n        }\n      }\n    }\n  }\n}\n'
GH_GET_TEAM_MEMBERS_QUERY = '\nquery($org: String!, $name: String!, $cursor: String) {\n  organization(login: $org) {\n    team(slug: $name) {\n      members(first: 100, after: $cursor) {\n        nodes {\n          login\n        }\n        pageInfo {\n          hasNextPage\n          endCursor\n        }\n      }\n    }\n  }\n}\n'
GH_GET_PR_NEXT_AUTHORS_QUERY = GH_COMMIT_AUTHORS_FRAGMENT + '\nquery ($owner: String!, $name: String!, $number: Int!, $cursor: String) {\n  repository(name: $name, owner: $owner) {\n    pullRequest(number: $number) {\n      commits_with_authors: commits(first: 100, after: $cursor) {\n        ...CommitAuthors\n      }\n    }\n  }\n}\n'
GH_GET_PR_PREV_REVIEWS_QUERY = GH_PR_REVIEWS_FRAGMENT + '\nquery ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {\n  repository(name: $name, owner: $owner) {\n    pullRequest(number: $number) {\n      reviews(last: 100, before: $cursor) {\n        ...PRReviews\n      }\n    }\n  }\n}\n'
GH_GET_REPO_SUBMODULES = '\nquery ($owner: String!, $name: String!) {\n  repository(owner: $owner, name: $name) {\n    submodules(first: 100) {\n      nodes {\n        path\n      }\n      pageInfo {\n        endCursor\n        hasNextPage\n      }\n    }\n  }\n}\n'
RE_GHSTACK_HEAD_REF = re.compile('^(gh/[^/]+/[0-9]+/)head$')
RE_GHSTACK_DESC = re.compile('Stack.*:\\r?\\n(\\* [^\\r\\n]+\\r?\\n)+', re.MULTILINE)
RE_PULL_REQUEST_RESOLVED = re.compile('Pull Request resolved: https://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>[0-9]+)', re.MULTILINE)
RE_PR_CC_LINE = re.compile('^cc:? @\\w+.*\\r?\\n?$', re.MULTILINE)
RE_DIFF_REV = re.compile('^Differential Revision:.+?(D[0-9]+)', re.MULTILINE)
CIFLOW_LABEL = re.compile('^ciflow/.+')
CIFLOW_TRUNK_LABEL = re.compile('^ciflow/trunk')
MERGE_RULE_PATH = Path('.github') / 'merge_rules.yaml'
ROCKSET_MERGES_COLLECTION = 'merges'
ROCKSET_MERGES_WORKSPACE = 'commons'
REMOTE_MAIN_BRANCH = 'origin/main'
DRCI_CHECKRUN_NAME = 'Dr.CI'
INTERNAL_CHANGES_CHECKRUN_NAME = 'Meta Internal-Only Changes Check'
HAS_NO_CONNECTED_DIFF_TITLE = 'There is no internal Diff connected, this can be merged now'
IGNORABLE_FAILED_CHECKS_THESHOLD = 10

def gh_graphql(query: str, **kwargs: Any) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    rc = gh_fetch_url('https://api.github.com/graphql', data={'query': query, 'variables': kwargs}, reader=json.load)
    if 'errors' in rc:
        raise RuntimeError(f"GraphQL query {query}, args {kwargs} failed: {rc['errors']}")
    return cast(Dict[str, Any], rc)

def gh_get_pr_info(org: str, proj: str, pr_no: int) -> Any:
    if False:
        print('Hello World!')
    rc = gh_graphql(GH_GET_PR_INFO_QUERY, name=proj, owner=org, number=pr_no)
    return rc['data']['repository']['pullRequest']

@lru_cache(maxsize=None)
def gh_get_team_members(org: str, name: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    rc: List[str] = []
    team_members: Dict[str, Any] = {'pageInfo': {'hasNextPage': 'true', 'endCursor': None}}
    while bool(team_members['pageInfo']['hasNextPage']):
        query = gh_graphql(GH_GET_TEAM_MEMBERS_QUERY, org=org, name=name, cursor=team_members['pageInfo']['endCursor'])
        team = query['data']['organization']['team']
        if team is None:
            warn(f'Requested non-existing team {org}/{name}')
            return []
        team_members = team['members']
        rc += [member['login'] for member in team_members['nodes']]
    return rc

def get_check_run_name_prefix(workflow_run: Any) -> str:
    if False:
        for i in range(10):
            print('nop')
    if workflow_run is None:
        return ''
    else:
        return f"{workflow_run['workflow']['name']} / "

def is_passing_status(status: Optional[str]) -> bool:
    if False:
        while True:
            i = 10
    return status is not None and status.upper() in ['SUCCESS', 'SKIPPED', 'NEUTRAL']

def add_workflow_conclusions(checksuites: Any, get_next_checkruns_page: Callable[[List[Dict[str, Dict[str, Any]]], int, Any], Any], get_next_checksuites: Callable[[Any], Any]) -> JobNameToStateDict:
    if False:
        print('Hello World!')
    workflows: Dict[str, WorkflowCheckState] = {}
    no_workflow_obj: WorkflowCheckState = WorkflowCheckState('', '', None)

    def add_conclusions(edges: Any) -> None:
        if False:
            i = 10
            return i + 15
        for (edge_idx, edge) in enumerate(edges):
            node = edge['node']
            workflow_run = node['workflowRun']
            checkruns = node['checkRuns']
            workflow_obj: WorkflowCheckState = no_workflow_obj
            if workflow_run is not None:
                workflow_name = workflow_run['workflow']['name']
                workflow_conclusion = node['conclusion']
                if workflow_conclusion == 'CANCELLED' and workflow_name in workflows:
                    continue
                if workflow_name not in workflows:
                    workflows[workflow_name] = WorkflowCheckState(name=workflow_name, status=workflow_conclusion, url=workflow_run['url'])
                workflow_obj = workflows[workflow_name]
            while checkruns is not None:
                for checkrun_node in checkruns['nodes']:
                    if not isinstance(checkrun_node, dict):
                        warn(f'Expected dictionary, but got {type(checkrun_node)}')
                        continue
                    checkrun_name = f"{get_check_run_name_prefix(workflow_run)}{checkrun_node['name']}"
                    existing_checkrun = workflow_obj.jobs.get(checkrun_name)
                    if existing_checkrun is None or not is_passing_status(existing_checkrun.status):
                        workflow_obj.jobs[checkrun_name] = JobCheckState(checkrun_name, checkrun_node['detailsUrl'], checkrun_node['conclusion'], classification=None, job_id=checkrun_node['databaseId'], title=checkrun_node['title'], summary=checkrun_node['summary'])
                if bool(checkruns['pageInfo']['hasNextPage']):
                    checkruns = get_next_checkruns_page(edges, edge_idx, checkruns)
                else:
                    checkruns = None
    all_edges = checksuites['edges'].copy()
    while bool(checksuites['pageInfo']['hasNextPage']):
        checksuites = get_next_checksuites(checksuites)
        all_edges.extend(checksuites['edges'])
    add_conclusions(all_edges)
    res: JobNameToStateDict = {}
    for (workflow_name, workflow) in workflows.items():
        if len(workflow.jobs) > 0:
            for (job_name, job) in workflow.jobs.items():
                res[job_name] = job
        else:
            res[workflow_name] = JobCheckState(workflow.name, workflow.url, workflow.status, classification=None, job_id=None, title=None, summary=None)
    for (job_name, job) in no_workflow_obj.jobs.items():
        res[job_name] = job
    return res

def parse_args() -> Any:
    if False:
        while True:
            i = 10
    from argparse import ArgumentParser
    parser = ArgumentParser('Merge PR into default branch')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--revert', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--ignore-current', action='store_true')
    parser.add_argument('--comment-id', type=int)
    parser.add_argument('--reason', type=str)
    parser.add_argument('pr_num', type=int)
    return parser.parse_args()

def can_skip_internal_checks(pr: 'GitHubPR', comment_id: Optional[int]=None) -> bool:
    if False:
        i = 10
        return i + 15
    if comment_id is None:
        return False
    comment = pr.get_comment_by_id(comment_id)
    if comment.editor_login is not None:
        return False
    return comment.author_login == 'facebook-github-bot'

def get_ghstack_prs(repo: GitRepo, pr: 'GitHubPR', open_only: bool=True) -> List[Tuple['GitHubPR', str]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the PRs in the stack that are below this PR (inclusive).  Throws error if any of the open PRs are out of sync.\n    @:param open_only: Only return open PRs\n    '
    assert pr.is_ghstack_pr()
    entire_stack: List[Tuple[GitHubPR, str]] = []
    orig_ref = f"{repo.remote}/{re.sub('/head$', '/orig', pr.head_ref())}"
    rev_list = repo.revlist(f'{pr.default_branch()}..{orig_ref}')
    for (idx, rev) in enumerate(reversed(rev_list)):
        msg = repo.commit_message(rev)
        m = RE_PULL_REQUEST_RESOLVED.search(msg)
        if m is None:
            raise RuntimeError(f'Could not find PR-resolved string in {msg} of ghstacked PR {pr.pr_num}')
        if pr.org != m.group('owner') or pr.project != m.group('repo'):
            raise RuntimeError(f"PR {m.group('number')} resolved to wrong owner/repo pair")
        stacked_pr_num = int(m.group('number'))
        if stacked_pr_num != pr.pr_num:
            stacked_pr = GitHubPR(pr.org, pr.project, stacked_pr_num)
            if open_only and stacked_pr.is_closed():
                print(f'Skipping {idx + 1} of {len(rev_list)} PR (#{stacked_pr_num}) as its already been merged')
                continue
            entire_stack.append((stacked_pr, rev))
        else:
            entire_stack.append((pr, rev))
    for (stacked_pr, rev) in entire_stack:
        if stacked_pr.is_closed():
            continue
        if not are_ghstack_branches_in_sync(repo, stacked_pr.head_ref()):
            raise RuntimeError(f'PR {stacked_pr.pr_num} is out of sync with the corresponding revision {rev} on ' + f'branch {orig_ref} that would be merged into main.  ' + 'This usually happens because there is a non ghstack change in the PR.  ' + f'Please sync them and try again (ex. make the changes on {orig_ref} and run ghstack).')
    return entire_stack

class GitHubPR:

    def __init__(self, org: str, project: str, pr_num: int) -> None:
        if False:
            return 10
        assert isinstance(pr_num, int)
        self.org = org
        self.project = project
        self.pr_num = pr_num
        self.info = gh_get_pr_info(org, project, pr_num)
        self.changed_files: Optional[List[str]] = None
        self.labels: Optional[List[str]] = None
        self.conclusions: Optional[JobNameToStateDict] = None
        self.comments: Optional[List[GitHubComment]] = None
        self._authors: Optional[List[Tuple[str, str]]] = None
        self._reviews: Optional[List[Tuple[str, str]]] = None
        self.merge_base: Optional[str] = None
        self.submodules: Optional[List[str]] = None

    def is_closed(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return bool(self.info['closed'])

    def is_cross_repo(self) -> bool:
        if False:
            return 10
        return bool(self.info['isCrossRepository'])

    def base_ref(self) -> str:
        if False:
            i = 10
            return i + 15
        return cast(str, self.info['baseRefName'])

    def default_branch(self) -> str:
        if False:
            i = 10
            return i + 15
        return cast(str, self.info['baseRepository']['defaultBranchRef']['name'])

    def head_ref(self) -> str:
        if False:
            return 10
        return cast(str, self.info['headRefName'])

    def is_ghstack_pr(self) -> bool:
        if False:
            i = 10
            return i + 15
        return RE_GHSTACK_HEAD_REF.match(self.head_ref()) is not None

    def is_base_repo_private(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return bool(self.info['baseRepository']['isPrivate'])

    def get_changed_files_count(self) -> int:
        if False:
            return 10
        return int(self.info['changedFiles'])

    def last_commit(self) -> Any:
        if False:
            return 10
        return self.info['commits']['nodes'][-1]['commit']

    def get_merge_base(self) -> str:
        if False:
            while True:
                i = 10
        if self.merge_base:
            return self.merge_base
        last_commit_oid = self.last_commit()['oid']
        self.merge_base = gh_fetch_merge_base(self.org, self.project, last_commit_oid, 'main')
        if not self.merge_base:
            self.merge_base = cast(str, self.info['baseRefOid'])
        return self.merge_base

    def get_changed_files(self) -> List[str]:
        if False:
            return 10
        if self.changed_files is None:
            info = self.info
            unique_changed_files = set()
            for _ in range(100):
                unique_changed_files.update([x['path'] for x in info['files']['nodes']])
                if not info['files']['pageInfo']['hasNextPage']:
                    break
                rc = gh_graphql(GH_GET_PR_NEXT_FILES_QUERY, name=self.project, owner=self.org, number=self.pr_num, cursor=info['files']['pageInfo']['endCursor'])
                info = rc['data']['repository']['pullRequest']
            self.changed_files = list(unique_changed_files)
        if len(self.changed_files) != self.get_changed_files_count():
            raise RuntimeError('Changed file count mismatch')
        return self.changed_files

    def get_submodules(self) -> List[str]:
        if False:
            return 10
        if self.submodules is None:
            rc = gh_graphql(GH_GET_REPO_SUBMODULES, name=self.project, owner=self.org)
            info = rc['data']['repository']['submodules']
            self.submodules = [s['path'] for s in info['nodes']]
        return self.submodules

    def get_changed_submodules(self) -> List[str]:
        if False:
            print('Hello World!')
        submodules = self.get_submodules()
        return [f for f in self.get_changed_files() if f in submodules]

    def has_invalid_submodule_updates(self) -> bool:
        if False:
            print('Hello World!')
        'Submodule updates in PR are invalid if submodule keyword\n        is not mentioned in neither the title nor body/description\n        nor in any of the labels.\n        '
        return len(self.get_changed_submodules()) > 0 and 'submodule' not in self.get_title().lower() and ('submodule' not in self.get_body().lower()) and all(('submodule' not in label for label in self.get_labels()))

    def _get_reviews(self) -> List[Tuple[str, str]]:
        if False:
            print('Hello World!')
        if self._reviews is None:
            self._reviews = []
            info = self.info
            for _ in range(100):
                nodes = info['reviews']['nodes']
                self._reviews = [(node['author']['login'], node['state']) for node in nodes] + self._reviews
                if not info['reviews']['pageInfo']['hasPreviousPage']:
                    break
                rc = gh_graphql(GH_GET_PR_PREV_REVIEWS_QUERY, name=self.project, owner=self.org, number=self.pr_num, cursor=info['reviews']['pageInfo']['startCursor'])
                info = rc['data']['repository']['pullRequest']
        reviews = {}
        for (author, state) in self._reviews:
            if state != 'COMMENTED':
                reviews[author] = state
        return list(reviews.items())

    def get_approved_by(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return [login for (login, state) in self._get_reviews() if state == 'APPROVED']

    def get_commit_count(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return int(self.info['commits_with_authors']['totalCount'])

    def get_pr_creator_login(self) -> str:
        if False:
            return 10
        return cast(str, self.info['author']['login'])

    def _fetch_authors(self) -> List[Tuple[str, str]]:
        if False:
            while True:
                i = 10
        if self._authors is not None:
            return self._authors
        authors: List[Tuple[str, str]] = []

        def add_authors(info: Dict[str, Any]) -> None:
            if False:
                i = 10
                return i + 15
            for node in info['commits_with_authors']['nodes']:
                author_node = node['commit']['author']
                user_node = author_node['user']
                author = f"{author_node['name']} <{author_node['email']}>"
                if user_node is None:
                    authors.append(('', author))
                else:
                    authors.append((cast(str, user_node['login']), author))
        info = self.info
        for _ in range(100):
            add_authors(info)
            if not info['commits_with_authors']['pageInfo']['hasNextPage']:
                break
            rc = gh_graphql(GH_GET_PR_NEXT_AUTHORS_QUERY, name=self.project, owner=self.org, number=self.pr_num, cursor=info['commits_with_authors']['pageInfo']['endCursor'])
            info = rc['data']['repository']['pullRequest']
        self._authors = authors
        return authors

    def get_committer_login(self, num: int=0) -> str:
        if False:
            return 10
        return self._fetch_authors()[num][0]

    def get_committer_author(self, num: int=0) -> str:
        if False:
            while True:
                i = 10
        return self._fetch_authors()[num][1]

    def get_labels(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        if self.labels is not None:
            return self.labels
        labels = [node['node']['name'] for node in self.info['labels']['edges']] if 'labels' in self.info else []
        self.labels = labels
        return self.labels

    def get_checkrun_conclusions(self) -> JobNameToStateDict:
        if False:
            return 10
        'Returns dict of checkrun -> [conclusion, url]'
        if self.conclusions is not None:
            return self.conclusions
        orig_last_commit = self.last_commit()

        def get_pr_next_check_runs(edges: List[Dict[str, Dict[str, Any]]], edge_idx: int, checkruns: Any) -> Any:
            if False:
                print('Hello World!')
            rc = gh_graphql(GH_GET_PR_NEXT_CHECK_RUNS, name=self.project, owner=self.org, number=self.pr_num, cs_cursor=edges[edge_idx - 1]['cursor'] if edge_idx > 0 else None, cr_cursor=checkruns['pageInfo']['endCursor'])
            last_commit = rc['data']['repository']['pullRequest']['commits']['nodes'][-1]['commit']
            checkruns = last_commit['checkSuites']['nodes'][-1]['checkRuns']
            return checkruns

        def get_pr_next_checksuites(checksuites: Any) -> Any:
            if False:
                print('Hello World!')
            rc = gh_graphql(GH_GET_PR_NEXT_CHECKSUITES, name=self.project, owner=self.org, number=self.pr_num, cursor=checksuites['edges'][-1]['cursor'])
            info = rc['data']['repository']['pullRequest']
            last_commit = info['commits']['nodes'][-1]['commit']
            if last_commit['oid'] != orig_last_commit['oid']:
                raise RuntimeError('Last commit changed on PR')
            return last_commit['checkSuites']
        checksuites = orig_last_commit['checkSuites']
        self.conclusions = add_workflow_conclusions(checksuites, get_pr_next_check_runs, get_pr_next_checksuites)
        if orig_last_commit['status'] and orig_last_commit['status']['contexts']:
            for status in orig_last_commit['status']['contexts']:
                name = status['context']
                self.conclusions[name] = JobCheckState(name, status['targetUrl'], status['state'], classification=None, job_id=None, title=None, summary=None)
        return self.conclusions

    def get_authors(self) -> Dict[str, str]:
        if False:
            return 10
        rc = {}
        if self.get_commit_count() <= 250:
            assert len(self._fetch_authors()) == self.get_commit_count()
        for idx in range(len(self._fetch_authors())):
            rc[self.get_committer_login(idx)] = self.get_committer_author(idx)
        return rc

    def get_author(self) -> str:
        if False:
            print('Hello World!')
        authors = self.get_authors()
        if len(authors) == 1:
            return next(iter(authors.values()))
        creator = self.get_pr_creator_login()
        if creator not in authors:
            return self.get_committer_author(0)
        return authors[creator]

    def get_title(self) -> str:
        if False:
            return 10
        return cast(str, self.info['title'])

    def get_body(self) -> str:
        if False:
            return 10
        return cast(str, self.info['body'])

    def get_merge_commit(self) -> Optional[str]:
        if False:
            print('Hello World!')
        mc = self.info['mergeCommit']
        return mc['oid'] if mc is not None else None

    def get_pr_url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'https://github.com/{self.org}/{self.project}/pull/{self.pr_num}'

    @staticmethod
    def _comment_from_node(node: Any) -> GitHubComment:
        if False:
            print('Hello World!')
        editor = node['editor']
        return GitHubComment(body_text=node['bodyText'], created_at=node['createdAt'] if 'createdAt' in node else '', author_login=node['author']['login'], author_association=node['authorAssociation'], editor_login=editor['login'] if editor else None, database_id=node['databaseId'], url=node['url'])

    def get_comments(self) -> List[GitHubComment]:
        if False:
            for i in range(10):
                print('nop')
        if self.comments is not None:
            return self.comments
        self.comments = []
        info = self.info['comments']
        for _ in range(100):
            self.comments = [self._comment_from_node(node) for node in info['nodes']] + self.comments
            if not info['pageInfo']['hasPreviousPage']:
                break
            rc = gh_graphql(GH_GET_PR_PREV_COMMENTS, name=self.project, owner=self.org, number=self.pr_num, cursor=info['pageInfo']['startCursor'])
            info = rc['data']['repository']['pullRequest']['comments']
        return self.comments

    def get_last_comment(self) -> GitHubComment:
        if False:
            for i in range(10):
                print('nop')
        return self._comment_from_node(self.info['comments']['nodes'][-1])

    def get_comment_by_id(self, database_id: int) -> GitHubComment:
        if False:
            print('Hello World!')
        if self.comments is None:
            for node in self.info['comments']['nodes']:
                comment = self._comment_from_node(node)
                if comment.database_id == database_id:
                    return comment
        for comment in self.get_comments():
            if comment.database_id == database_id:
                return comment
        for node in self.info['reviews']['nodes']:
            comment = self._comment_from_node(node)
            if comment.database_id == database_id:
                return comment
        raise RuntimeError(f'Comment with id {database_id} not found')

    def get_diff_revision(self) -> Optional[str]:
        if False:
            return 10
        rc = RE_DIFF_REV.search(self.get_body())
        return rc.group(1) if rc is not None else None

    def has_internal_changes(self) -> bool:
        if False:
            while True:
                i = 10
        checkrun_name = INTERNAL_CHANGES_CHECKRUN_NAME
        if self.get_diff_revision() is None:
            return False
        checks = self.get_checkrun_conclusions()
        if checks is None or checkrun_name not in checks:
            return False
        return checks[checkrun_name].status != 'SUCCESS'

    def has_no_connected_diff(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        checkrun_name = INTERNAL_CHANGES_CHECKRUN_NAME
        checks = self.get_checkrun_conclusions()
        if checks is None or checkrun_name not in checks:
            return False
        return checks[checkrun_name].title == HAS_NO_CONNECTED_DIFF_TITLE

    def merge_ghstack_into(self, repo: GitRepo, skip_mandatory_checks: bool, comment_id: Optional[int]=None) -> List['GitHubPR']:
        if False:
            for i in range(10):
                print('nop')
        assert self.is_ghstack_pr()
        ghstack_prs = get_ghstack_prs(repo, self, open_only=False)
        pr_dependencies = []
        for (pr, rev) in ghstack_prs:
            if pr.is_closed():
                pr_dependencies.append(pr)
                continue
            commit_msg = pr.gen_commit_message(filter_ghstack=True, ghstack_deps=pr_dependencies)
            if pr.pr_num != self.pr_num:
                find_matching_merge_rule(pr, repo, skip_mandatory_checks=skip_mandatory_checks, skip_internal_checks=can_skip_internal_checks(self, comment_id))
            repo.cherry_pick(rev)
            repo.amend_commit_message(commit_msg)
            pr_dependencies.append(pr)
        return [x for (x, _) in ghstack_prs if not x.is_closed()]

    def gen_commit_message(self, filter_ghstack: bool=False, ghstack_deps: Optional[List['GitHubPR']]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Fetches title and body from PR description\n        adds reviewed by, pull request resolved and optionally\n        filters out ghstack info'
        approved_by_urls = ', '.join((prefix_with_github_url(login) for login in self.get_approved_by()))
        msg_body = re.sub(RE_PR_CC_LINE, '', self.get_body())
        if filter_ghstack:
            msg_body = re.sub(RE_GHSTACK_DESC, '', msg_body)
        msg = self.get_title() + f' (#{self.pr_num})\n\n'
        msg += msg_body
        msg += f'\nPull Request resolved: {self.get_pr_url()}\n'
        msg += f'Approved by: {approved_by_urls}\n'
        if ghstack_deps:
            msg += f"ghstack dependencies: {', '.join([f'#{pr.pr_num}' for pr in ghstack_deps])}\n"
        return msg

    def add_numbered_label(self, label_base: str) -> None:
        if False:
            i = 10
            return i + 15
        labels = self.get_labels() if self.labels is not None else []
        full_label = label_base
        count = 0
        for label in labels:
            if label_base in label:
                count += 1
                full_label = f'{label_base}X{count}'
        gh_add_labels(self.org, self.project, self.pr_num, [full_label])

    def merge_into(self, repo: GitRepo, *, skip_mandatory_checks: bool=False, dry_run: bool=False, comment_id: Optional[int]=None, ignore_current_checks: Optional[List[str]]=None) -> None:
        if False:
            while True:
                i = 10
        (merge_rule, pending_checks, failed_checks, ignorable_checks) = find_matching_merge_rule(self, repo, skip_mandatory_checks=skip_mandatory_checks, skip_internal_checks=can_skip_internal_checks(self, comment_id), ignore_current_checks=ignore_current_checks)
        additional_merged_prs = self.merge_changes(repo, skip_mandatory_checks, comment_id)
        repo.push(self.default_branch(), dry_run)
        if not dry_run:
            self.add_numbered_label(MERGE_COMPLETE_LABEL)
            for pr in additional_merged_prs:
                pr.add_numbered_label(MERGE_COMPLETE_LABEL)
        if comment_id and self.pr_num:
            merge_commit_sha = repo.rev_parse(name=REMOTE_MAIN_BRANCH)
            save_merge_record(collection=ROCKSET_MERGES_COLLECTION, comment_id=comment_id, pr_num=self.pr_num, owner=self.org, project=self.project, author=self.get_author(), pending_checks=pending_checks, failed_checks=failed_checks, ignore_current_checks=ignorable_checks.get('IGNORE_CURRENT_CHECK', []), broken_trunk_checks=ignorable_checks.get('BROKEN_TRUNK', []), flaky_checks=ignorable_checks.get('FLAKY', []), unstable_checks=ignorable_checks.get('UNSTABLE', []), last_commit_sha=self.last_commit().get('oid', ''), merge_base_sha=self.get_merge_base(), merge_commit_sha=merge_commit_sha, is_failed=False, dry_run=dry_run, skip_mandatory_checks=skip_mandatory_checks, ignore_current=bool(ignore_current_checks), workspace=ROCKSET_MERGES_WORKSPACE)
        else:
            print("Missing comment ID or PR number, couldn't upload to Rockset")

    def merge_changes(self, repo: GitRepo, skip_mandatory_checks: bool=False, comment_id: Optional[int]=None, branch: Optional[str]=None) -> List['GitHubPR']:
        if False:
            for i in range(10):
                print('nop')
        branch_to_merge_into = self.default_branch() if branch is None else branch
        if repo.current_branch() != branch_to_merge_into:
            repo.checkout(branch_to_merge_into)
        if not self.is_ghstack_pr():
            msg = self.gen_commit_message()
            pr_branch_name = f'__pull-request-{self.pr_num}__init__'
            repo.fetch(f'pull/{self.pr_num}/head', pr_branch_name)
            repo._run_git('merge', '--squash', pr_branch_name)
            repo._run_git('commit', f'--author="{self.get_author()}"', '-m', msg)
            return []
        else:
            return self.merge_ghstack_into(repo, skip_mandatory_checks, comment_id=comment_id)

class MergeRuleFailedError(RuntimeError):

    def __init__(self, message: str, rule: Optional['MergeRule']=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(message)
        self.rule = rule

class MandatoryChecksMissingError(MergeRuleFailedError):
    pass

class PostCommentError(Exception):
    pass

@dataclass
class MergeRule:
    name: str
    patterns: List[str]
    approved_by: List[str]
    mandatory_checks_name: Optional[List[str]]
    ignore_flaky_failures: bool = True

def gen_new_issue_link(org: str, project: str, labels: List[str], template: str='bug-report.yml') -> str:
    if False:
        while True:
            i = 10
    labels_str = ','.join(labels)
    return f'https://github.com/{org}/{project}/issues/new?labels={urllib.parse.quote(labels_str)}&template={urllib.parse.quote(template)}'

def read_merge_rules(repo: Optional[GitRepo], org: str, project: str) -> List[MergeRule]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the list of all merge rules for the repo or project.\n\n    NB: this function is used in Meta-internal workflows, see the comment\n    at the top of this file for details.\n    '
    repo_relative_rules_path = MERGE_RULE_PATH
    if repo is None:
        json_data = gh_fetch_url(f'https://api.github.com/repos/{org}/{project}/contents/{repo_relative_rules_path}', headers={'Accept': 'application/vnd.github.v3+json'}, reader=json.load)
        content = base64.b64decode(json_data['content'])
        return [MergeRule(**x) for x in yaml.safe_load(content)]
    else:
        rules_path = Path(repo.repo_dir) / repo_relative_rules_path
        if not rules_path.exists():
            print(f'{rules_path} does not exist, returning empty rules')
            return []
        with open(rules_path) as fp:
            rc = yaml.safe_load(fp)
        return [MergeRule(**x) for x in rc]

def find_matching_merge_rule(pr: GitHubPR, repo: Optional[GitRepo]=None, skip_mandatory_checks: bool=False, skip_internal_checks: bool=False, ignore_current_checks: Optional[List[str]]=None) -> Tuple[MergeRule, List[Tuple[str, Optional[str], Optional[int]]], List[Tuple[str, Optional[str], Optional[int]]], Dict[str, List[Any]]]:
    if False:
        return 10
    '\n    Returns merge rule matching to this pr together with the list of associated pending\n    and failing jobs OR raises an exception.\n\n    NB: this function is used in Meta-internal workflows, see the comment at the top of\n    this file for details.\n    '
    changed_files = pr.get_changed_files()
    approved_by = set(pr.get_approved_by())
    issue_link = gen_new_issue_link(org=pr.org, project=pr.project, labels=['module: ci'])
    reject_reason = f'No rule found to match PR. Please [report]{issue_link} this issue to DevX team.'
    rules = read_merge_rules(repo, pr.org, pr.project)
    if not rules:
        reject_reason = f'Rejecting the merge as no rules are defined for the repository in {MERGE_RULE_PATH}'
        raise RuntimeError(reject_reason)
    checks = pr.get_checkrun_conclusions()
    checks = get_classifications(pr.pr_num, pr.project, checks, ignore_current_checks=ignore_current_checks)
    reject_reason_score = 0
    for rule in rules:
        rule_name = rule.name
        patterns_re = patterns_to_regex(rule.patterns)
        non_matching_files = []
        for fname in changed_files:
            if not patterns_re.match(fname):
                non_matching_files.append(fname)
        if len(non_matching_files) > 0:
            num_matching_files = len(changed_files) - len(non_matching_files)
            if num_matching_files > reject_reason_score:
                reject_reason_score = num_matching_files
                reject_reason = '\n'.join((f'Not all files match rule `{rule_name}`.', f'{num_matching_files} files matched, but there are still non-matching files:', f"{','.join(non_matching_files[:5])}{(', ...' if len(non_matching_files) > 5 else '')}"))
            continue
        if len(rule.approved_by) > 0 and len(approved_by) == 0:
            if reject_reason_score < 10000:
                reject_reason_score = 10000
                reject_reason = f'PR #{pr.pr_num} has not been reviewed yet'
            continue
        rule_approvers_set = set()
        for approver in rule.approved_by:
            if '/' in approver:
                (org, name) = approver.split('/')
                rule_approvers_set.update(gh_get_team_members(org, name))
            else:
                rule_approvers_set.add(approver)
        approvers_intersection = approved_by.intersection(rule_approvers_set)
        if len(approvers_intersection) == 0 and len(rule_approvers_set) > 0:
            if reject_reason_score < 10000:
                reject_reason_score = 10000
                reject_reason = '\n'.join(('Approval needed from one of the following:', f"{', '.join(list(rule_approvers_set)[:5])}{(', ...' if len(rule_approvers_set) > 5 else '')}"))
            continue
        mandatory_checks = rule.mandatory_checks_name if rule.mandatory_checks_name is not None else []
        required_checks = list(filter(lambda x: 'EasyCLA' in x or not skip_mandatory_checks, mandatory_checks))
        (pending_checks, failed_checks, _) = categorize_checks(checks, required_checks, ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD if rule.ignore_flaky_failures else 0)
        hud_link = f"https://hud.pytorch.org/{pr.org}/{pr.project}/commit/{pr.last_commit()['oid']}"
        if len(failed_checks) > 0:
            if reject_reason_score < 30000:
                reject_reason_score = 30000
                reject_reason = '\n'.join((f'{len(failed_checks)} mandatory check(s) failed.  The first few are:', *checks_to_markdown_bullets(failed_checks), '', f'Dig deeper by [viewing the failures on hud]({hud_link})'))
            continue
        elif len(pending_checks) > 0:
            if reject_reason_score < 20000:
                reject_reason_score = 20000
                reject_reason = '\n'.join((f'{len(pending_checks)} mandatory check(s) are pending/not yet run.  The first few are:', *checks_to_markdown_bullets(pending_checks), '', f'Dig deeper by [viewing the pending checks on hud]({hud_link})'))
            continue
        if not skip_internal_checks and pr.has_internal_changes():
            raise RuntimeError('This PR has internal changes and must be landed via Phabricator')
        (pending_mandatory_checks, failed_mandatory_checks, ignorable_checks) = categorize_checks(checks, [], ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD)
        return (rule, pending_mandatory_checks, failed_mandatory_checks, ignorable_checks)
    if reject_reason_score == 20000:
        raise MandatoryChecksMissingError(reject_reason, rule)
    raise MergeRuleFailedError(reject_reason, rule)

def checks_to_str(checks: List[Tuple[str, Optional[str]]]) -> str:
    if False:
        i = 10
        return i + 15
    return ', '.join((f'[{c[0]}]({c[1]})' if c[1] is not None else c[0] for c in checks))

def checks_to_markdown_bullets(checks: List[Tuple[str, Optional[str], Optional[int]]]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    return [f'- [{c[0]}]({c[1]})' if c[1] is not None else f'- {c[0]}' for c in checks[:5]]

@retries_decorator()
def save_merge_record(collection: str, comment_id: int, pr_num: int, owner: str, project: str, author: str, pending_checks: List[Tuple[str, Optional[str], Optional[int]]], failed_checks: List[Tuple[str, Optional[str], Optional[int]]], ignore_current_checks: List[Tuple[str, Optional[str], Optional[int]]], broken_trunk_checks: List[Tuple[str, Optional[str], Optional[int]]], flaky_checks: List[Tuple[str, Optional[str], Optional[int]]], unstable_checks: List[Tuple[str, Optional[str], Optional[int]]], last_commit_sha: str, merge_base_sha: str, merge_commit_sha: str='', is_failed: bool=False, dry_run: bool=False, skip_mandatory_checks: bool=False, ignore_current: bool=False, error: str='', workspace: str='commons') -> None:
    if False:
        while True:
            i = 10
    '\n    This saves the merge records into Rockset, so we can query them (for fun and profit)\n    '
    if dry_run:
        return
    try:
        import rockset
        data = [{'comment_id': comment_id, 'pr_num': pr_num, 'owner': owner, 'project': project, 'author': author, 'pending_checks': pending_checks, 'failed_checks': failed_checks, 'ignore_current_checks': ignore_current_checks, 'broken_trunk_checks': broken_trunk_checks, 'flaky_checks': flaky_checks, 'unstable_checks': unstable_checks, 'last_commit_sha': last_commit_sha, 'merge_base_sha': merge_base_sha, 'merge_commit_sha': merge_commit_sha, 'is_failed': is_failed, 'skip_mandatory_checks': skip_mandatory_checks, 'ignore_current': ignore_current, 'error': error}]
        client = rockset.RocksetClient(host='api.usw2a1.rockset.com', api_key=os.environ['ROCKSET_API_KEY'])
        client.Documents.add_documents(collection=collection, data=data, workspace=workspace)
    except ModuleNotFoundError:
        print('Rockset is missing, no record will be saved')
        return

@retries_decorator(rc=[])
def get_rockset_results(head_sha: str, merge_base: str) -> List[Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    query = f"\nSELECT\n    w.name as workflow_name,\n    j.id,\n    j.name,\n    j.conclusion,\n    j.completed_at,\n    j.html_url,\n    j.head_sha,\n    j.torchci_classification.captures as failure_captures,\n    LENGTH(j.steps) as steps,\nFROM\n    commons.workflow_job j join commons.workflow_run w on w.id = j.run_id\nwhere\n    j.head_sha in ('{head_sha}','{merge_base}')\n"
    try:
        import rockset
        res = rockset.RocksetClient(host='api.usw2a1.rockset.com', api_key=os.environ['ROCKSET_API_KEY']).sql(query)
        return cast(List[Dict[str, Any]], res.results)
    except ModuleNotFoundError:
        print('Could not use RockSet as rocket dependency is missing')
        return []

@retries_decorator()
def get_drci_classifications(pr_num: int, project: str='pytorch') -> Any:
    if False:
        while True:
            i = 10
    '\n    Query HUD API to find similar failures to decide if they are flaky\n    '
    failures = gh_fetch_url(f'https://hud.pytorch.org/api/drci/drci?prNumber={pr_num}', data=f'repo={project}', headers={'Authorization': os.getenv('DRCI_BOT_KEY', ''), 'Accept': 'application/vnd.github.v3+json'}, method='POST', reader=json.load)
    return failures.get(str(pr_num), {}) if failures else {}
REMOVE_JOB_NAME_SUFFIX_REGEX = re.compile(', [0-9]+, [0-9]+, .+\\)$')

def remove_job_name_suffix(name: str, replacement: str=')') -> str:
    if False:
        for i in range(10):
            print('nop')
    return re.sub(REMOVE_JOB_NAME_SUFFIX_REGEX, replacement, name)

def is_broken_trunk(name: str, drci_classifications: Any) -> bool:
    if False:
        return 10
    if not name or not drci_classifications:
        return False
    return any((name == broken_trunk['name'] for broken_trunk in drci_classifications.get('BROKEN_TRUNK', [])))

def is_flaky(name: str, drci_classifications: Any) -> bool:
    if False:
        print('Hello World!')
    if not name or not drci_classifications:
        return False
    return any((name == flaky['name'] for flaky in drci_classifications.get('FLAKY', [])))

def is_invalid_cancel(name: str, conclusion: Optional[str], drci_classifications: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    After https://github.com/pytorch/test-infra/pull/4579, invalid cancelled\n    signals have been removed from HUD and Dr.CI. The same needs to be done\n    here for consistency\n    '
    if not name or not drci_classifications or (not conclusion) or (conclusion.upper() != 'CANCELLED'):
        return False
    return all((name != failure['name'] for failure in drci_classifications.get('FAILED', [])))

def get_classifications(pr_num: int, project: str, checks: Dict[str, JobCheckState], ignore_current_checks: Optional[List[str]]) -> Dict[str, JobCheckState]:
    if False:
        return 10
    drci_classifications = get_drci_classifications(pr_num=pr_num, project=project)
    print(f'From Dr.CI API: {json.dumps(drci_classifications)}')
    if not drci_classifications and DRCI_CHECKRUN_NAME in checks and checks[DRCI_CHECKRUN_NAME] and checks[DRCI_CHECKRUN_NAME].summary:
        drci_summary = checks[DRCI_CHECKRUN_NAME].summary
        try:
            print(f'From Dr.CI checkrun summary: {drci_summary}')
            drci_classifications = json.loads(str(drci_summary))
        except json.JSONDecodeError as error:
            warn('Invalid Dr.CI checkrun summary')
            drci_classifications = {}
    checks_with_classifications = checks.copy()
    for (name, check) in checks.items():
        if check.status == 'SUCCESS' or check.status == 'NEUTRAL':
            continue
        if 'unstable' in name:
            checks_with_classifications[name] = JobCheckState(check.name, check.url, check.status, 'UNSTABLE', check.job_id, check.title, check.summary)
            continue
        if is_broken_trunk(name, drci_classifications):
            checks_with_classifications[name] = JobCheckState(check.name, check.url, check.status, 'BROKEN_TRUNK', check.job_id, check.title, check.summary)
            continue
        elif is_flaky(name, drci_classifications):
            checks_with_classifications[name] = JobCheckState(check.name, check.url, check.status, 'FLAKY', check.job_id, check.title, check.summary)
            continue
        elif is_invalid_cancel(name, check.status, drci_classifications):
            checks_with_classifications[name] = JobCheckState(check.name, check.url, check.status, 'INVALID_CANCEL', check.job_id, check.title, check.summary)
            continue
        if ignore_current_checks is not None and name in ignore_current_checks:
            checks_with_classifications[name] = JobCheckState(check.name, check.url, check.status, 'IGNORE_CURRENT_CHECK', check.job_id, check.title, check.summary)
    return checks_with_classifications

def filter_checks_with_lambda(checks: JobNameToStateDict, status_filter: Callable[[Optional[str]], bool]) -> List[JobCheckState]:
    if False:
        return 10
    return [check for check in checks.values() if status_filter(check.status)]

def validate_revert(repo: GitRepo, pr: GitHubPR, *, comment_id: Optional[int]=None) -> Tuple[str, str]:
    if False:
        while True:
            i = 10
    comment = pr.get_last_comment() if comment_id is None else pr.get_comment_by_id(comment_id)
    if comment.editor_login is not None:
        raise PostCommentError("Don't want to revert based on edited command")
    author_association = comment.author_association
    author_login = comment.author_login
    allowed_reverters = ['COLLABORATOR', 'MEMBER', 'OWNER']
    if pr.is_base_repo_private():
        allowed_reverters.append('CONTRIBUTOR')
    if author_association not in allowed_reverters:
        raise PostCommentError(f"Will not revert as @{author_login} is not one of [{', '.join(allowed_reverters)}], but instead is {author_association}.")
    skip_internal_checks = can_skip_internal_checks(pr, comment_id)
    if pr.has_no_connected_diff():
        skip_internal_checks = True
    find_matching_merge_rule(pr, repo, skip_mandatory_checks=True, skip_internal_checks=skip_internal_checks)
    commit_sha = pr.get_merge_commit()
    if commit_sha is None:
        commits = repo.commits_resolving_gh_pr(pr.pr_num)
        if len(commits) == 0:
            raise PostCommentError("Can't find any commits resolving PR")
        commit_sha = commits[0]
    msg = repo.commit_message(commit_sha)
    rc = RE_DIFF_REV.search(msg)
    if rc is not None and (not skip_internal_checks):
        raise PostCommentError(f"Can't revert PR that was landed via phabricator as {rc.group(1)}.  " + 'Please revert by going to the internal diff and clicking Unland.')
    return (author_login, commit_sha)

def try_revert(repo: GitRepo, pr: GitHubPR, *, dry_run: bool=False, comment_id: Optional[int]=None, reason: Optional[str]=None) -> None:
    if False:
        i = 10
        return i + 15

    def post_comment(msg: str) -> None:
        if False:
            while True:
                i = 10
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, msg, dry_run=dry_run)
    try:
        (author_login, commit_sha) = validate_revert(repo, pr, comment_id=comment_id)
    except PostCommentError as e:
        return post_comment(str(e))
    revert_msg = f'\nReverted {pr.get_pr_url()} on behalf of {prefix_with_github_url(author_login)}'
    revert_msg += f' due to {reason}' if reason is not None else ''
    revert_msg += f' ([comment]({pr.get_comment_by_id(comment_id).url}))\n' if comment_id is not None else '\n'
    repo.checkout(pr.default_branch())
    repo.revert(commit_sha)
    msg = repo.commit_message('HEAD')
    msg = re.sub(RE_PULL_REQUEST_RESOLVED, '', msg)
    msg += revert_msg
    repo.amend_commit_message(msg)
    repo.push(pr.default_branch(), dry_run)
    post_comment(f'@{pr.get_pr_creator_login()} your PR has been successfully reverted.')
    if not dry_run:
        pr.add_numbered_label('reverted')
        gh_post_commit_comment(pr.org, pr.project, commit_sha, revert_msg)
        gh_update_pr_state(pr.org, pr.project, pr.pr_num)

def prefix_with_github_url(suffix_str: str) -> str:
    if False:
        return 10
    return f'https://github.com/{suffix_str}'

def check_for_sev(org: str, project: str, skip_mandatory_checks: bool) -> None:
    if False:
        print('Hello World!')
    if skip_mandatory_checks:
        return
    response = cast(Dict[str, Any], gh_fetch_json_list('https://api.github.com/search/issues', params={'q': f'repo:{org}/{project} is:open is:issue label:"ci: sev"'}))
    if response['total_count'] != 0:
        for item in response['items']:
            if 'MERGE BLOCKING' in item['body']:
                raise RuntimeError('Not merging any PRs at the moment because there is a ' + 'merge blocking https://github.com/pytorch/pytorch/labels/ci:%20sev issue open at: \n' + f"{item['html_url']}")
    return

def has_label(labels: List[str], pattern: Pattern[str]=CIFLOW_LABEL) -> bool:
    if False:
        print('Hello World!')
    return len(list(filter(pattern.match, labels))) > 0

def categorize_checks(check_runs: JobNameToStateDict, required_checks: List[str], ok_failed_checks_threshold: Optional[int]=None) -> Tuple[List[Tuple[str, Optional[str], Optional[int]]], List[Tuple[str, Optional[str], Optional[int]]], Dict[str, List[Any]]]:
    if False:
        while True:
            i = 10
    '\n    Categories all jobs into the list of pending and failing jobs. All known flaky\n    failures and broken trunk are ignored by defaults when ok_failed_checks_threshold\n    is not set (unlimited)\n    '
    pending_checks: List[Tuple[str, Optional[str], Optional[int]]] = []
    failed_checks: List[Tuple[str, Optional[str], Optional[int]]] = []
    ok_failed_checks: List[Tuple[str, Optional[str], Optional[int]]] = []
    ignorable_failed_checks: Dict[str, List[Any]] = defaultdict(list)
    relevant_checknames = [name for name in check_runs.keys() if not required_checks or any((x in name for x in required_checks))]
    for checkname in required_checks:
        if all((checkname not in x for x in check_runs.keys())):
            pending_checks.append((checkname, None, None))
    for checkname in relevant_checknames:
        status = check_runs[checkname].status
        url = check_runs[checkname].url
        classification = check_runs[checkname].classification
        job_id = check_runs[checkname].job_id
        if status is None and classification != 'UNSTABLE':
            pending_checks.append((checkname, url, job_id))
        elif classification == 'INVALID_CANCEL':
            continue
        elif not is_passing_status(check_runs[checkname].status):
            target = ignorable_failed_checks[classification] if classification in ('IGNORE_CURRENT_CHECK', 'BROKEN_TRUNK', 'FLAKY', 'UNSTABLE') else failed_checks
            target.append((checkname, url, job_id))
            if classification in ('BROKEN_TRUNK', 'FLAKY', 'UNSTABLE'):
                ok_failed_checks.append((checkname, url, job_id))
    if ok_failed_checks:
        warn(f'The following {len(ok_failed_checks)} checks failed but were likely due flakiness or broken trunk: ' + ', '.join([x[0] for x in ok_failed_checks]) + (f' but this is greater than the threshold of {ok_failed_checks_threshold} so merge will fail' if ok_failed_checks_threshold is not None and len(ok_failed_checks) > ok_failed_checks_threshold else ''))
    if ok_failed_checks_threshold is not None and len(ok_failed_checks) > ok_failed_checks_threshold:
        failed_checks = failed_checks + ok_failed_checks
    return (pending_checks, failed_checks, ignorable_failed_checks)

def merge(pr: GitHubPR, repo: GitRepo, dry_run: bool=False, skip_mandatory_checks: bool=False, comment_id: Optional[int]=None, timeout_minutes: int=400, stale_pr_days: int=3, ignore_current: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    initial_commit_sha = pr.last_commit()['oid']
    pr_link = f'https://github.com/{pr.org}/{pr.project}/pull/{pr.pr_num}'
    print(f'Attempting merge of {initial_commit_sha} ({pr_link})')
    if MERGE_IN_PROGRESS_LABEL not in pr.get_labels():
        gh_add_labels(pr.org, pr.project, pr.pr_num, [MERGE_IN_PROGRESS_LABEL])
    explainer = TryMergeExplainer(skip_mandatory_checks, pr.get_labels(), pr.pr_num, pr.org, pr.project, ignore_current)
    ignore_current_checks_info = []
    if pr.is_ghstack_pr():
        get_ghstack_prs(repo, pr)
    check_for_sev(pr.org, pr.project, skip_mandatory_checks)
    if skip_mandatory_checks or can_skip_internal_checks(pr, comment_id):
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, explainer.get_merge_message(), dry_run=dry_run)
        return pr.merge_into(repo, dry_run=dry_run, skip_mandatory_checks=skip_mandatory_checks, comment_id=comment_id)
    find_matching_merge_rule(pr, repo, skip_mandatory_checks=True)
    if not has_required_labels(pr):
        raise RuntimeError(LABEL_ERR_MSG.lstrip(' #'))
    if ignore_current:
        checks = pr.get_checkrun_conclusions()
        (_, failing, _) = categorize_checks(checks, list(checks.keys()), ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD)
        ignore_current_checks_info = failing
    gh_post_pr_comment(pr.org, pr.project, pr.pr_num, explainer.get_merge_message(ignore_current_checks_info), dry_run=dry_run)
    start_time = time.time()
    last_exception = ''
    elapsed_time = 0.0
    ignore_current_checks = [x[0] for x in ignore_current_checks_info]
    while elapsed_time < timeout_minutes * 60:
        check_for_sev(pr.org, pr.project, skip_mandatory_checks)
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(f'Attempting merge of https://github.com/{pr.org}/{pr.project}/pull/{pr.pr_num} ({elapsed_time / 60} minutes elapsed)')
        pr = GitHubPR(pr.org, pr.project, pr.pr_num)
        if initial_commit_sha != pr.last_commit()['oid']:
            raise RuntimeError('New commits were pushed while merging. Please rerun the merge command.')
        try:
            required_checks = []
            failed_rule_message = None
            ignore_flaky_failures = True
            try:
                find_matching_merge_rule(pr, repo, ignore_current_checks=ignore_current_checks)
            except MandatoryChecksMissingError as ex:
                if ex.rule is not None:
                    ignore_flaky_failures = ex.rule.ignore_flaky_failures
                    if ex.rule.mandatory_checks_name is not None:
                        required_checks = ex.rule.mandatory_checks_name
                failed_rule_message = ex
            checks = pr.get_checkrun_conclusions()
            checks = get_classifications(pr.pr_num, pr.project, checks, ignore_current_checks=ignore_current_checks)
            (pending, failing, _) = categorize_checks(checks, required_checks + [x for x in checks.keys() if x not in required_checks], ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD if ignore_flaky_failures else 0)
            startup_failures = filter_checks_with_lambda(checks, lambda status: status == 'STARTUP_FAILURE')
            if len(startup_failures) > 0:
                raise RuntimeError(f'{len(startup_failures)} STARTUP failures reported, please check workflows syntax! ' + ', '.join((f'[{x.name}]({x.url})' for x in startup_failures[:5])))
            if len(failing) > 0:
                raise RuntimeError(f'{len(failing)} jobs have failed, first few of them are: ' + ', '.join((f'[{x[0]}]({x[1]})' for x in failing[:5])))
            if len(pending) > 0:
                if failed_rule_message is not None:
                    raise failed_rule_message
                else:
                    raise MandatoryChecksMissingError(f'Still waiting for {len(pending)} jobs to finish, ' + f"first few of them are: {', '.join((x[0] for x in pending[:5]))}")
            return pr.merge_into(repo, dry_run=dry_run, skip_mandatory_checks=skip_mandatory_checks, comment_id=comment_id, ignore_current_checks=ignore_current_checks)
        except MandatoryChecksMissingError as ex:
            last_exception = str(ex)
            print(f'Merge of https://github.com/{pr.org}/{pr.project}/pull/{pr.pr_num} failed due to: {ex}. Retrying in 5 min')
            time.sleep(5 * 60)
    msg = f'Merged timed out after {timeout_minutes} minutes. Please contact the pytorch_dev_infra team.'
    msg += f'The last exception was: {last_exception}'
    if not dry_run:
        gh_add_labels(pr.org, pr.project, pr.pr_num, ['land-failed'])
    raise RuntimeError(msg)

def main() -> None:
    if False:
        i = 10
        return i + 15
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    (org, project) = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    def handle_exception(e: Exception, title: str='Merge failed') -> None:
        if False:
            return 10
        exception = f'**Reason**: {e}'
        failing_rule = None
        if isinstance(e, MergeRuleFailedError):
            failing_rule = e.rule.name if e.rule else None
        internal_debugging = ''
        run_url = os.getenv('GH_RUN_URL')
        if run_url is not None:
            internal_debugging = '\n'.join((line for line in ('<details><summary>Details for Dev Infra team</summary>', f'Raised by <a href="{run_url}">workflow job</a>\n', f'Failing merge rule: {failing_rule}' if failing_rule else '', '</details>') if line))
        msg = '\n'.join((f'## {title}', f'{exception}', '', f'{internal_debugging}'))
        gh_post_pr_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)
        import traceback
        traceback.print_exc()
    if args.revert:
        try:
            gh_post_pr_comment(org, project, args.pr_num, get_revert_message(org, project, pr.pr_num), args.dry_run)
            try_revert(repo, pr, dry_run=args.dry_run, comment_id=args.comment_id, reason=args.reason)
        except Exception as e:
            handle_exception(e, f'Reverting PR {args.pr_num} failed')
        return
    if pr.is_closed():
        gh_post_pr_comment(org, project, args.pr_num, f"Can't merge closed PR #{args.pr_num}", dry_run=args.dry_run)
        return
    if pr.is_cross_repo() and pr.is_ghstack_pr():
        gh_post_pr_comment(org, project, args.pr_num, 'Cross-repo ghstack merges are not supported', dry_run=args.dry_run)
        return
    if not args.force and pr.has_invalid_submodule_updates():
        message = f"This PR updates submodules {', '.join(pr.get_changed_submodules())}\n"
        message += '\nIf those updates are intentional, please add "submodule" keyword to PR title/description.'
        gh_post_pr_comment(org, project, args.pr_num, message, dry_run=args.dry_run)
        return
    try:
        merge(pr, repo, dry_run=args.dry_run, skip_mandatory_checks=args.force, comment_id=args.comment_id, ignore_current=args.ignore_current)
    except Exception as e:
        handle_exception(e)
        if args.comment_id and args.pr_num:
            save_merge_record(collection=ROCKSET_MERGES_COLLECTION, comment_id=args.comment_id, pr_num=args.pr_num, owner=org, project=project, author=pr.get_author(), pending_checks=[], failed_checks=[], ignore_current_checks=[], broken_trunk_checks=[], flaky_checks=[], unstable_checks=[], last_commit_sha=pr.last_commit().get('oid', ''), merge_base_sha=pr.get_merge_base(), is_failed=True, dry_run=args.dry_run, skip_mandatory_checks=args.force, ignore_current=args.ignore_current, error=str(e), workspace=ROCKSET_MERGES_WORKSPACE)
        else:
            print("Missing comment ID or PR number, couldn't upload to Rockset")
    finally:
        gh_remove_label(org, project, args.pr_num, MERGE_IN_PROGRESS_LABEL)
if __name__ == '__main__':
    main()