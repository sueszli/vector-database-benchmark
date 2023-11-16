"""Bot triggered by Github Actions every time a new issue, PR or comment
is created. Assign labels, provide replies, closes issues, etc. depending
on the situation.
"""
from __future__ import print_function
import functools
import json
import os
import re
from pprint import pprint as pp
from github import Github
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
LABELS_MAP = {'linux': ['linux', 'ubuntu', 'redhat', 'mint', 'centos', 'red hat', 'archlinux', 'debian', 'alpine', 'gentoo', 'fedora', 'slackware', 'suse', 'RHEL', 'opensuse', 'manylinux', 'apt ', 'apt-', 'rpm', 'yum', 'kali', '/sys/class', '/proc/net', '/proc/disk', '/proc/smaps', '/proc/vmstat'], 'windows': ['windows', 'win32', 'WinError', 'WindowsError', 'win10', 'win7', 'win ', 'mingw', 'msys', 'studio', 'microsoft', 'make.bat', 'CloseHandle', 'GetLastError', 'NtQuery', 'DLL', 'MSVC', 'TCHAR', 'WCHAR', '.bat', 'OpenProcess', 'TerminateProcess', 'appveyor', 'windows error', 'NtWow64', 'NTSTATUS', 'Visual Studio'], 'macos': ['macos', 'mac ', 'osx', 'os x', 'mojave', 'sierra', 'capitan', 'yosemite', 'catalina', 'mojave', 'big sur', 'xcode', 'darwin', 'dylib', 'm1'], 'aix': ['aix'], 'cygwin': ['cygwin'], 'freebsd': ['freebsd'], 'netbsd': ['netbsd'], 'openbsd': ['openbsd'], 'sunos': ['sunos', 'solaris'], 'wsl': ['wsl'], 'unix': ['psposix', '_psutil_posix', 'waitpid', 'statvfs', '/dev/tty', '/dev/pts', 'posix'], 'pypy': ['pypy'], 'enhancement': ['enhancement'], 'memleak': ['memory leak', 'leaks memory', 'memleak', 'mem leak'], 'api': ['idea', 'proposal', 'api', 'feature'], 'performance': ['performance', 'speedup', 'speed up', 'slow', 'fast'], 'wheels': ['wheel', 'wheels'], 'scripts': ['example script', 'examples script', 'example dir', 'scripts/'], 'bug': ['fail', "can't execute", "can't install", 'cannot execute', 'cannot install', 'install error', 'crash', 'critical'], 'doc': ['doc ', 'document ', 'documentation', 'readthedocs', 'pythonhosted', 'HISTORY', 'README', 'dev guide', 'devguide', 'sphinx', 'docfix', 'index.rst'], 'tests': [' test ', 'tests', 'travis', 'coverage', 'cirrus', 'appveyor', 'continuous integration', 'unittest', 'pytest', 'unit test'], 'priority-high': ['WinError', 'WindowsError', 'RuntimeError', 'ZeroDivisionError', 'SystemError', 'MemoryError', 'core dumped', 'segfault', 'segmentation fault']}
LABELS_MAP['scripts'].extend([x for x in os.listdir(SCRIPTS_DIR) if x.endswith('.py')])
OS_LABELS = ['linux', 'windows', 'macos', 'freebsd', 'openbsd', 'netbsd', 'openbsd', 'bsd', 'sunos', 'unix', 'wsl', 'aix', 'cygwin']
ILLOGICAL_PAIRS = [('bug', 'enhancement'), ('doc', 'tests'), ('scripts', 'doc'), ('scripts', 'tests'), ('bsd', 'freebsd'), ('bsd', 'openbsd'), ('bsd', 'netbsd')]
REPLY_MISSING_PYTHON_HEADERS = "It looks like you're missing `Python.h` headers. This usually means you have to install them first, then retry psutil installation.\nPlease read [INSTALL](https://github.com/giampaolo/psutil/blob/master/INSTALL.rst) instructions for your platform. This is an auto-generated response based on the text you submitted. If this was a mistake or you think there's a bug with psutil installation process, please add a comment to reopen this issue.\n"

def is_pr(issue):
    if False:
        return 10
    return issue.pull_request is not None

def has_label(issue, label):
    if False:
        for i in range(10):
            print('nop')
    assigned = [x.name for x in issue.labels]
    return label in assigned

def has_os_label(issue):
    if False:
        while True:
            i = 10
    labels = set([x.name for x in issue.labels])
    return any((x in labels for x in OS_LABELS))

def get_repo():
    if False:
        for i in range(10):
            print('nop')
    repo = os.environ['GITHUB_REPOSITORY']
    token = os.environ['GITHUB_TOKEN']
    return Github(token).get_repo(repo)

@functools.lru_cache()
def _get_event_data():
    if False:
        i = 10
        return i + 15
    ret = json.load(open(os.environ['GITHUB_EVENT_PATH']))
    pp(ret)
    return ret

def is_event_new_issue():
    if False:
        i = 10
        return i + 15
    data = _get_event_data()
    try:
        return data['action'] == 'opened' and 'issue' in data
    except KeyError:
        return False

def is_event_new_pr():
    if False:
        return 10
    data = _get_event_data()
    try:
        return data['action'] == 'opened' and 'pull_request' in data
    except KeyError:
        return False

def get_issue():
    if False:
        for i in range(10):
            print('nop')
    data = _get_event_data()
    try:
        num = data['issue']['number']
    except KeyError:
        num = data['pull_request']['number']
    return get_repo().get_issue(number=num)

def log(msg):
    if False:
        i = 10
        return i + 15
    if '\n' in msg or '\r\n' in msg:
        print('>>>\n%s\n<<<' % msg, flush=True)
    else:
        print('>>> %s <<<' % msg, flush=True)

def add_label(issue, label):
    if False:
        for i in range(10):
            print('nop')

    def should_add(issue, label):
        if False:
            print('Hello World!')
        if has_label(issue, label):
            log('already has label %r' % label)
            return False
        for (left, right) in ILLOGICAL_PAIRS:
            if label == left and has_label(issue, right):
                log('already has label' % label)
                return False
        return not has_label(issue, label)
    if not should_add(issue, label):
        log('should not add label %r' % label)
        return
    log('add label %r' % label)
    issue.add_to_labels(label)

def _guess_labels_from_text(text):
    if False:
        i = 10
        return i + 15
    assert isinstance(text, str), text
    for (label, keywords) in LABELS_MAP.items():
        for keyword in keywords:
            if keyword.lower() in text.lower():
                yield (label, keyword)

def add_labels_from_text(issue, text):
    if False:
        print('Hello World!')
    assert isinstance(text, str), text
    for (label, keyword) in _guess_labels_from_text(text):
        add_label(issue, label)

def add_labels_from_new_body(issue, text):
    if False:
        while True:
            i = 10
    assert isinstance(text, str), text
    log('start searching for template lines in new issue/PR body')
    r = re.search('\\* OS:.*?\\n', text)
    log("search for 'OS: ...' line")
    if r:
        log('found')
        add_labels_from_text(issue, r.group(0))
    else:
        log('not found')
    log("search for 'Bug fix: y/n' line")
    r = re.search('\\* Bug fix:.*?\\n', text)
    if is_pr(issue) and r is not None and (not has_label(issue, 'bug')) and (not has_label(issue, 'enhancement')):
        log('found')
        s = r.group(0).lower()
        if 'yes' in s:
            add_label(issue, 'bug')
        else:
            add_label(issue, 'enhancement')
    else:
        log('not found')
    log("search for 'Type: ...' line")
    r = re.search('\\* Type:.*?\\n', text)
    if r:
        log('found')
        s = r.group(0).lower()
        if 'doc' in s:
            add_label(issue, 'doc')
        if 'performance' in s:
            add_label(issue, 'performance')
        if 'scripts' in s:
            add_label(issue, 'scripts')
        if 'tests' in s:
            add_label(issue, 'tests')
        if 'wheels' in s:
            add_label(issue, 'wheels')
        if 'new-api' in s:
            add_label(issue, 'new-api')
        if 'new-platform' in s:
            add_label(issue, 'new-platform')
    else:
        log('not found')

def on_new_issue(issue):
    if False:
        for i in range(10):
            print('nop')

    def has_text(text):
        if False:
            for i in range(10):
                print('nop')
        return text in issue.title.lower() or (issue.body and text in issue.body.lower())

    def body_mentions_python_h():
        if False:
            return 10
        if not issue.body:
            return False
        body = issue.body.replace(' ', '')
        return '#include<Python.h>\n^~~~' in body or '#include<Python.h>\r\n^~~~' in body
    log('searching for missing Python.h')
    if has_text('missing python.h') or has_text('python.h: no such file or directory') or body_mentions_python_h():
        log('found mention of Python.h')
        issue.create_comment(REPLY_MISSING_PYTHON_HEADERS)
        issue.edit(state='closed')
        return

def on_new_pr(issue):
    if False:
        print('Hello World!')
    pass

def main():
    if False:
        while True:
            i = 10
    issue = get_issue()
    stype = 'PR' if is_pr(issue) else 'issue'
    log('running issue bot for %s %r' % (stype, issue))
    if is_event_new_issue():
        log('created new issue %s' % issue)
        add_labels_from_text(issue, issue.title)
        if issue.body:
            add_labels_from_new_body(issue, issue.body)
        on_new_issue(issue)
    elif is_event_new_pr():
        log('created new PR %s' % issue)
        add_labels_from_text(issue, issue.title)
        if issue.body:
            add_labels_from_new_body(issue, issue.body)
        on_new_pr(issue)
    else:
        log('unhandled event')
if __name__ == '__main__':
    main()