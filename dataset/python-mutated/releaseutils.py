import re
import sys
from subprocess import Popen, PIPE
try:
    from jira.client import JIRA
    try:
        from jira.exceptions import JIRAError
    except ImportError:
        from jira.utils import JIRAError
except ImportError:
    print('This tool requires the jira-python library')
    print("Install using 'pip3 install jira'")
    sys.exit(-1)
try:
    from github import Github
    from github import GithubException
except ImportError:
    print('This tool requires the PyGithub library')
    print("Install using 'pip install PyGithub'")
    sys.exit(-1)
contributors_file_name = 'contributors.txt'

def yesOrNoPrompt(msg):
    if False:
        for i in range(10):
            print('nop')
    response = input('%s [y/n]: ' % msg)
    while response != 'y' and response != 'n':
        return yesOrNoPrompt(msg)
    return response == 'y'

def run_cmd(cmd):
    if False:
        return 10
    return Popen(cmd, stdout=PIPE).communicate()[0].decode('utf8')

def run_cmd_error(cmd):
    if False:
        i = 10
        return i + 15
    return Popen(cmd, stdout=PIPE, stderr=PIPE).communicate()[1].decode('utf8')

def get_date(commit_hash):
    if False:
        i = 10
        return i + 15
    return run_cmd(['git', 'show', '--quiet', '--pretty=format:%cd', commit_hash])

def tag_exists(tag):
    if False:
        print('Hello World!')
    stderr = run_cmd_error(['git', 'show', tag])
    return 'error' not in stderr

class Commit:

    def __init__(self, _hash, author, title, pr_number=None):
        if False:
            return 10
        self._hash = _hash
        self.author = author
        self.title = title
        self.pr_number = pr_number

    def get_hash(self):
        if False:
            print('Hello World!')
        return self._hash

    def get_author(self):
        if False:
            for i in range(10):
                print('nop')
        return self.author

    def get_title(self):
        if False:
            while True:
                i = 10
        return self.title

    def get_pr_number(self):
        if False:
            while True:
                i = 10
        return self.pr_number

    def __str__(self):
        if False:
            print('Hello World!')
        closes_pr = '(Closes #%s)' % self.pr_number if self.pr_number else ''
        return '%s %s %s %s' % (self._hash, self.author, self.title, closes_pr)

def get_commits(tag):
    if False:
        i = 10
        return i + 15
    commit_start_marker = '|=== COMMIT START MARKER ===|'
    commit_end_marker = '|=== COMMIT END MARKER ===|'
    field_end_marker = '|=== COMMIT FIELD END MARKER ===|'
    log_format = commit_start_marker + '%h' + field_end_marker + '%an' + field_end_marker + '%s' + commit_end_marker + '%b'
    output = run_cmd(['git', 'log', '--quiet', '--pretty=format:' + log_format, tag])
    commits = []
    raw_commits = [c for c in output.split(commit_start_marker) if c]
    for commit in raw_commits:
        if commit.count(commit_end_marker) != 1:
            print('Commit end marker not found in commit: ')
            for line in commit.split('\n'):
                print(line)
            sys.exit(1)
        [commit_digest, commit_body] = commit.split(commit_end_marker)
        if commit_digest.count(field_end_marker) != 2:
            sys.exit('Unexpected format in commit: %s' % commit_digest)
        [_hash, author, title] = commit_digest.split(field_end_marker)
        pr_number = None
        match = re.search('Closes #([0-9]+) from ([^/\\s]+)/', commit_body)
        if match:
            [pr_number, github_username] = match.groups()
            if not is_valid_author(author):
                author = github_username
        author = author.strip()
        commit = Commit(_hash, author, title, pr_number)
        commits.append(commit)
    return commits
known_issue_types = {'bug': 'bug fixes', 'build': 'build fixes', 'dependency upgrade': 'build fixes', 'improvement': 'improvements', 'new feature': 'new features', 'documentation': 'documentation', 'test': 'test', 'task': 'improvement', 'sub-task': 'improvement'}
CORE_COMPONENT = 'Core'
known_components = {'block manager': CORE_COMPONENT, 'build': CORE_COMPONENT, 'deploy': CORE_COMPONENT, 'documentation': CORE_COMPONENT, 'examples': CORE_COMPONENT, 'graphx': 'GraphX', 'input/output': CORE_COMPONENT, 'java api': 'Java API', 'k8s': 'Kubernetes', 'kubernetes': 'Kubernetes', 'ml': 'MLlib', 'mllib': 'MLlib', 'project infra': 'Project Infra', 'pyspark': 'PySpark', 'shuffle': 'Shuffle', 'spark core': CORE_COMPONENT, 'spark shell': CORE_COMPONENT, 'sql': 'SQL', 'streaming': 'Streaming', 'web ui': 'Web UI', 'windows': 'Windows', 'yarn': 'YARN'}

def translate_issue_type(issue_type, issue_id, warnings):
    if False:
        for i in range(10):
            print('nop')
    issue_type = issue_type.lower()
    if issue_type in known_issue_types:
        return known_issue_types[issue_type]
    else:
        warnings.append('Unknown issue type "%s" (see %s)' % (issue_type, issue_id))
        return issue_type

def translate_component(component, commit_hash, warnings):
    if False:
        print('Hello World!')
    component = component.lower()
    if component in known_components:
        return known_components[component]
    else:
        warnings.append('Unknown component "%s" (see %s)' % (component, commit_hash))
        return component

def find_components(commit, commit_hash):
    if False:
        for i in range(10):
            print('nop')
    components = re.findall('\\[\\w*\\]', commit.lower())
    components = [translate_component(c, commit_hash, []) for c in components if c in known_components]
    return components

def nice_join(str_list):
    if False:
        return 10
    str_list = list(str_list)
    if not str_list:
        return ''
    elif len(str_list) == 1:
        return next(iter(str_list))
    elif len(str_list) == 2:
        return ' and '.join(str_list)
    else:
        return ', '.join(str_list[:-1]) + ', and ' + str_list[-1]

def get_github_name(author, github_client):
    if False:
        i = 10
        return i + 15
    if github_client:
        try:
            return github_client.get_user(author).name
        except GithubException as e:
            if e.status != 404:
                raise e
    return None

def get_jira_name(author, jira_client):
    if False:
        print('Hello World!')
    if jira_client:
        try:
            return jira_client.user(author).displayName
        except JIRAError as e:
            if e.status_code != 404:
                raise e
    return None

def is_valid_author(author):
    if False:
        while True:
            i = 10
    if not author:
        return False
    return ' ' in author and (not re.findall('[0-9]', author))

def capitalize_author(author):
    if False:
        i = 10
        return i + 15
    if not author:
        return None
    words = author.split(' ')
    words = [w[0].capitalize() + w[1:] for w in words if w]
    return ' '.join(words)