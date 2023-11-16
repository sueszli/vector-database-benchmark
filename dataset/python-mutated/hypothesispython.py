import os
import re
import shutil
import subprocess
import sys
import requests
import hypothesistooling as tools
from hypothesistooling import releasemanagement as rm
PACKAGE_NAME = 'hypothesis-python'
HYPOTHESIS_PYTHON = tools.ROOT / PACKAGE_NAME
PYTHON_TAG_PREFIX = 'hypothesis-python-'
BASE_DIR = HYPOTHESIS_PYTHON
PYTHON_SRC = HYPOTHESIS_PYTHON / 'src'
PYTHON_TESTS = HYPOTHESIS_PYTHON / 'tests'
DOMAINS_LIST = PYTHON_SRC / 'hypothesis' / 'vendor' / 'tlds-alpha-by-domain.txt'
RELEASE_FILE = HYPOTHESIS_PYTHON / 'RELEASE.rst'
assert PYTHON_SRC.exists()
__version__ = None
__version_info__ = None
VERSION_FILE = os.path.join(PYTHON_SRC, 'hypothesis/version.py')
with open(VERSION_FILE, encoding='utf-8') as o:
    exec(o.read())
assert __version__ is not None
assert __version_info__ is not None

def has_release():
    if False:
        while True:
            i = 10
    return RELEASE_FILE.exists()

def parse_release_file():
    if False:
        while True:
            i = 10
    return rm.parse_release_file(RELEASE_FILE)

def has_source_changes():
    if False:
        while True:
            i = 10
    return tools.has_changes([PYTHON_SRC])

def build_docs(builder='html'):
    if False:
        print('Hello World!')
    tools.scripts.pip_tool('sphinx-build', '-n', '-W', '--keep-going', '-T', '-E', '-b', builder, 'docs', 'docs/_build/' + builder, cwd=HYPOTHESIS_PYTHON)
CHANGELOG_ANCHOR = re.compile('^\\.\\. _v\\d+\\.\\d+\\.\\d+:$', flags=re.MULTILINE)
CHANGELOG_BORDER = re.compile('^-+$', flags=re.MULTILINE)
CHANGELOG_HEADER = re.compile('^\\d+\\.\\d+\\.\\d+ - \\d\\d\\d\\d-\\d\\d-\\d\\d$', flags=re.M)

def update_changelog_and_version():
    if False:
        for i in range(10):
            print('nop')
    global __version_info__
    global __version__
    contents = changelog()
    assert '\r' not in contents
    lines = contents.split('\n')
    for (i, l) in enumerate(lines):
        if CHANGELOG_ANCHOR.match(l):
            assert CHANGELOG_BORDER.match(lines[i + 2]), repr(lines[i + 2])
            assert CHANGELOG_HEADER.match(lines[i + 3]), repr(lines[i + 3])
            assert CHANGELOG_BORDER.match(lines[i + 4]), repr(lines[i + 4])
            assert lines[i + 3].startswith(__version__), f'__version__={__version__!r}   lines[i + 3]={lines[i + 3]!r}'
            beginning = '\n'.join(lines[:i])
            rest = '\n'.join(lines[i:])
            assert f'{beginning}\n{rest}' == contents
            break
    (release_type, release_contents) = parse_release_file()
    (new_version_string, new_version_info) = rm.bump_version_info(__version_info__, release_type)
    __version_info__ = new_version_info
    __version__ = new_version_string
    if release_type == 'major':
        (major, _, _) = __version_info__
        old = f'Hypothesis {major - 1}.x'
        beginning = beginning.replace(old, f'Hypothesis {major}.x')
        rest = '\n'.join([old, len(old) * '=', '', rest])
    rm.replace_assignment(VERSION_FILE, '__version_info__', repr(new_version_info))
    heading_for_new_version = f'{new_version_string} - {rm.release_date_string()}'
    border_for_new_version = '-' * len(heading_for_new_version)
    new_changelog_parts = [beginning.strip(), '', f'.. _v{new_version_string}:', '', border_for_new_version, heading_for_new_version, border_for_new_version, '', release_contents, '', rest]
    CHANGELOG_FILE.write_text('\n'.join(new_changelog_parts), encoding='utf-8')
    before = 'since="RELEASEDAY"'
    after = before.replace('RELEASEDAY', rm.release_date_string())
    for (root, _, files) in os.walk(PYTHON_SRC):
        for fname in (os.path.join(root, f) for f in files if f.endswith('.py')):
            with open(fname, encoding='utf-8') as f:
                contents = f.read()
            if before in contents:
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write(contents.replace(before, after))
CHANGELOG_FILE = HYPOTHESIS_PYTHON / 'docs' / 'changes.rst'
DIST = HYPOTHESIS_PYTHON / 'dist'

def changelog():
    if False:
        i = 10
        return i + 15
    return CHANGELOG_FILE.read_text(encoding='utf-8')

def build_distribution():
    if False:
        return 10
    if os.path.exists(DIST):
        shutil.rmtree(DIST)
    subprocess.check_output([sys.executable, 'setup.py', 'sdist', 'bdist_wheel', '--dist-dir', DIST])

def upload_distribution():
    if False:
        i = 10
        return i + 15
    tools.assert_can_release()
    subprocess.check_call([sys.executable, '-m', 'twine', 'upload', '--skip-existing', '--username=__token__', os.path.join(DIST, '*')])
    build_docs(builder='text')
    textfile = os.path.join(HYPOTHESIS_PYTHON, 'docs', '_build', 'text', 'changes.txt')
    with open(textfile, encoding='utf-8') as f:
        lines = f.readlines()
    entries = [i for (i, l) in enumerate(lines) if CHANGELOG_HEADER.match(l)]
    anchor = current_version().replace('.', '-')
    changelog_body = ''.join(lines[entries[0] + 2:entries[1]]).strip() + f'\n\n*[The canonical version of these notes (with links) is on readthedocs.](https://hypothesis.readthedocs.io/en/latest/changes.html#v{anchor})*'
    resp = requests.post('https://api.github.com/repos/HypothesisWorks/hypothesis/releases', headers={'Accept': 'application/vnd.github+json', 'Authorization': f"Bearer: {os.environ['GH_TOKEN']}", 'X-GitHub-Api-Version': '2022-11-28'}, json={'tag_name': tag_name(), 'name': 'Hypothesis for Python - version ' + current_version(), 'body': changelog_body}, timeout=120)
    try:
        resp.raise_for_status()
    except Exception:
        import traceback
        traceback.print_exc()

def current_version():
    if False:
        while True:
            i = 10
    return __version__

def latest_version():
    if False:
        return 10
    versions = []
    for t in tools.tags():
        if t.startswith(PYTHON_TAG_PREFIX):
            t = t[len(PYTHON_TAG_PREFIX):]
        else:
            continue
        assert t == t.strip()
        parts = t.split('.')
        assert len(parts) == 3
        v = tuple(map(int, parts))
        versions.append((v, t))
    (_, latest) = max(versions)
    return latest

def tag_name():
    if False:
        for i in range(10):
            print('nop')
    return PYTHON_TAG_PREFIX + __version__

def get_autoupdate_message(domainlist_changed):
    if False:
        for i in range(10):
            print('nop')
    if domainlist_changed:
        return 'This patch updates our vendored `list of top-level domains <https://www.iana.org/domains/root/db>`__,\nwhich is used by the provisional :func:`~hypothesis.provisional.domains` strategy.\n'
    return 'This patch updates our autoformatting tools, improving our code style without any API changes.'