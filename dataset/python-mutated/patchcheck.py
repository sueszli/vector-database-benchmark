"""Check proposed changes for common issues."""
import re
import sys
import shutil
import os.path
import subprocess
import sysconfig
import reindent
import untabify
EXCLUDE_DIRS = [os.path.join('Modules', '_ctypes', 'libffi_osx'), os.path.join('Modules', '_ctypes', 'libffi_msvc'), os.path.join('Modules', '_decimal', 'libmpdec'), os.path.join('Modules', 'expat'), os.path.join('Modules', 'zlib')]
SRCDIR = sysconfig.get_config_var('srcdir')

def n_files_str(count):
    if False:
        for i in range(10):
            print('nop')
    "Return 'N file(s)' with the proper plurality on 'file'."
    return '{} file{}'.format(count, 's' if count != 1 else '')

def status(message, modal=False, info=None):
    if False:
        return 10
    'Decorator to output status info to stdout.'

    def decorated_fxn(fxn):
        if False:
            for i in range(10):
                print('nop')

        def call_fxn(*args, **kwargs):
            if False:
                return 10
            sys.stdout.write(message + ' ... ')
            sys.stdout.flush()
            result = fxn(*args, **kwargs)
            if not modal and (not info):
                print('done')
            elif info:
                print(info(result))
            else:
                print('yes' if result else 'NO')
            return result
        return call_fxn
    return decorated_fxn

def get_git_branch():
    if False:
        for i in range(10):
            print('nop')
    'Get the symbolic name for the current git branch'
    cmd = 'git rev-parse --abbrev-ref HEAD'.split()
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, cwd=SRCDIR, encoding='UTF-8')
    except subprocess.CalledProcessError:
        return None

def get_git_upstream_remote():
    if False:
        while True:
            i = 10
    'Get the remote name to use for upstream branches\n\n    Uses "upstream" if it exists, "origin" otherwise\n    '
    cmd = 'git remote get-url upstream'.split()
    try:
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL, cwd=SRCDIR, encoding='UTF-8')
    except subprocess.CalledProcessError:
        return 'origin'
    return 'upstream'

def get_git_remote_default_branch(remote_name):
    if False:
        for i in range(10):
            print('nop')
    "Get the name of the default branch for the given remote\n\n    It is typically called 'main', but may differ\n    "
    cmd = 'git remote show {}'.format(remote_name).split()
    env = os.environ.copy()
    env['LANG'] = 'C'
    try:
        remote_info = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, cwd=SRCDIR, encoding='UTF-8', env=env)
    except subprocess.CalledProcessError:
        return None
    for line in remote_info.splitlines():
        if 'HEAD branch:' in line:
            base_branch = line.split(':')[1].strip()
            return base_branch
    return None

@status('Getting base branch for PR', info=lambda x: x if x is not None else 'not a PR branch')
def get_base_branch():
    if False:
        i = 10
        return i + 15
    if not os.path.exists(os.path.join(SRCDIR, '.git')):
        return None
    upstream_remote = get_git_upstream_remote()
    version = sys.version_info
    if version.releaselevel == 'alpha':
        base_branch = get_git_remote_default_branch(upstream_remote)
    else:
        base_branch = '{0.major}.{0.minor}'.format(version)
    this_branch = get_git_branch()
    if this_branch is None or this_branch == base_branch:
        return None
    return upstream_remote + '/' + base_branch

@status('Getting the list of files that have been added/changed', info=lambda x: n_files_str(len(x)))
def changed_files(base_branch=None):
    if False:
        return 10
    'Get the list of changed or added files from git.'
    if os.path.exists(os.path.join(SRCDIR, '.git')):
        if base_branch:
            cmd = 'git diff --name-status ' + base_branch
        else:
            cmd = 'git status --porcelain'
        filenames = []
        with subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, cwd=SRCDIR) as st:
            for line in st.stdout:
                line = line.decode().rstrip()
                (status_text, filename) = line.split(maxsplit=1)
                status = set(status_text)
                if not status.intersection('MAU'):
                    continue
                if ' -> ' in filename:
                    filename = filename.split(' -> ', 2)[1].strip()
                filenames.append(filename)
    else:
        sys.exit('need a git checkout to get modified files')
    filenames2 = []
    for filename in filenames:
        filename = os.path.normpath(filename)
        if any((filename.startswith(path) for path in EXCLUDE_DIRS)):
            continue
        filenames2.append(filename)
    return filenames2

def report_modified_files(file_paths):
    if False:
        while True:
            i = 10
    count = len(file_paths)
    if count == 0:
        return n_files_str(count)
    else:
        lines = ['{}:'.format(n_files_str(count))]
        for path in file_paths:
            lines.append('  {}'.format(path))
        return '\n'.join(lines)

@status('Fixing Python file whitespace', info=report_modified_files)
def normalize_whitespace(file_paths):
    if False:
        for i in range(10):
            print('nop')
    'Make sure that the whitespace for .py files have been normalized.'
    reindent.makebackup = False
    fixed = [path for path in file_paths if path.endswith('.py') and reindent.check(os.path.join(SRCDIR, path))]
    return fixed

@status('Fixing C file whitespace', info=report_modified_files)
def normalize_c_whitespace(file_paths):
    if False:
        return 10
    'Report if any C files '
    fixed = []
    for path in file_paths:
        abspath = os.path.join(SRCDIR, path)
        with open(abspath, 'r') as f:
            if '\t' not in f.read():
                continue
        untabify.process(abspath, 8, verbose=False)
        fixed.append(path)
    return fixed
ws_re = re.compile(b'\\s+(\\r?\\n)$')

@status('Fixing docs whitespace', info=report_modified_files)
def normalize_docs_whitespace(file_paths):
    if False:
        return 10
    fixed = []
    for path in file_paths:
        abspath = os.path.join(SRCDIR, path)
        try:
            with open(abspath, 'rb') as f:
                lines = f.readlines()
            new_lines = [ws_re.sub(b'\\1', line) for line in lines]
            if new_lines != lines:
                shutil.copyfile(abspath, abspath + '.bak')
                with open(abspath, 'wb') as f:
                    f.writelines(new_lines)
                fixed.append(path)
        except Exception as err:
            print('Cannot fix %s: %s' % (path, err))
    return fixed

@status('Docs modified', modal=True)
def docs_modified(file_paths):
    if False:
        i = 10
        return i + 15
    'Report if any file in the Doc directory has been changed.'
    return bool(file_paths)

@status('Misc/ACKS updated', modal=True)
def credit_given(file_paths):
    if False:
        return 10
    'Check if Misc/ACKS has been changed.'
    return os.path.join('Misc', 'ACKS') in file_paths

@status('Misc/NEWS.d updated with `blurb`', modal=True)
def reported_news(file_paths):
    if False:
        for i in range(10):
            print('nop')
    'Check if Misc/NEWS.d has been changed.'
    return any((p.startswith(os.path.join('Misc', 'NEWS.d', 'next')) for p in file_paths))

@status('configure regenerated', modal=True, info=str)
def regenerated_configure(file_paths):
    if False:
        i = 10
        return i + 15
    'Check if configure has been regenerated.'
    if 'configure.ac' in file_paths:
        return 'yes' if 'configure' in file_paths else 'no'
    else:
        return 'not needed'

@status('pyconfig.h.in regenerated', modal=True, info=str)
def regenerated_pyconfig_h_in(file_paths):
    if False:
        print('Hello World!')
    'Check if pyconfig.h.in has been regenerated.'
    if 'configure.ac' in file_paths:
        return 'yes' if 'pyconfig.h.in' in file_paths else 'no'
    else:
        return 'not needed'

def travis(pull_request):
    if False:
        return 10
    if pull_request == 'false':
        print('Not a pull request; skipping')
        return
    base_branch = get_base_branch()
    file_paths = changed_files(base_branch)
    python_files = [fn for fn in file_paths if fn.endswith('.py')]
    c_files = [fn for fn in file_paths if fn.endswith(('.c', '.h'))]
    doc_files = [fn for fn in file_paths if fn.startswith('Doc') and fn.endswith(('.rst', '.inc'))]
    fixed = []
    fixed.extend(normalize_whitespace(python_files))
    fixed.extend(normalize_c_whitespace(c_files))
    fixed.extend(normalize_docs_whitespace(doc_files))
    if not fixed:
        print('No whitespace issues found')
    else:
        print(f'Please fix the {len(fixed)} file(s) with whitespace issues')
        print('(on UNIX you can run `make patchcheck` to make the fixes)')
        sys.exit(1)

def main():
    if False:
        i = 10
        return i + 15
    base_branch = get_base_branch()
    file_paths = changed_files(base_branch)
    python_files = [fn for fn in file_paths if fn.endswith('.py')]
    c_files = [fn for fn in file_paths if fn.endswith(('.c', '.h'))]
    doc_files = [fn for fn in file_paths if fn.startswith('Doc') and fn.endswith(('.rst', '.inc'))]
    misc_files = {p for p in file_paths if p.startswith('Misc')}
    normalize_whitespace(python_files)
    normalize_c_whitespace(c_files)
    normalize_docs_whitespace(doc_files)
    docs_modified(doc_files)
    credit_given(misc_files)
    reported_news(misc_files)
    regenerated_configure(file_paths)
    regenerated_pyconfig_h_in(file_paths)
    if python_files or c_files:
        end = ' and check for refleaks?' if c_files else '?'
        print()
        print('Did you run the test suite' + end)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--travis', help='Perform pass/fail checks')
    args = parser.parse_args()
    if args.travis:
        travis(args.travis)
    else:
        main()