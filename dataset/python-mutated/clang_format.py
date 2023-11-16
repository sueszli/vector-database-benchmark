"""
A script that provides:
1. Validates clang-format is the right version.
2. Has support for checking which files are to be checked.
3. Supports validating and updating a set of files to the right coding style.
"""
import queue
import difflib
import glob
import itertools
import os
import re
import subprocess
from subprocess import check_output, CalledProcessError
import sys
import threading
import time
import shutil
from argparse import ArgumentParser
from multiprocessing import cpu_count
if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__)))))
CLANG_FORMAT_VERSION = '7.0.1'
CLANG_FORMAT_SHORT_VERSION = '7.0'
CLANG_FORMAT_PROGNAME = 'clang-format'
files_match = re.compile('\\.(h|cc|c)$')

def callo(args):
    if False:
        return 10
    'Call a program, and capture its output\n    '
    return check_output(args).decode('utf-8')

class ClangFormat(object):
    """Class encapsulates finding a suitable copy of clang-format,
    and linting/formatting an individual file
    """

    def __init__(self, path):
        if False:
            while True:
                i = 10
        self.path = None
        clang_format_progname_ext = ''
        if sys.platform == 'win32':
            clang_format_progname_ext += '.exe'
        if path is not None:
            if os.path.isfile(path):
                self.path = path
            else:
                print('WARNING: Could not find clang-format %s' % path)
        if self.path is None:
            programs = [CLANG_FORMAT_PROGNAME + '-' + CLANG_FORMAT_VERSION, CLANG_FORMAT_PROGNAME + '-' + CLANG_FORMAT_SHORT_VERSION, CLANG_FORMAT_PROGNAME]
            if sys.platform == 'win32':
                for i in range(len(programs)):
                    programs[i] += '.exe'
            for program in programs:
                self.path = shutil.which(program)
                if self.path:
                    if not self._validate_version():
                        self.path = None
                    else:
                        break
        if sys.platform == 'win32':
            programfiles = [os.environ['ProgramFiles'], os.environ['ProgramFiles(x86)']]
            for programfile in programfiles:
                win32bin = os.path.join(programfile, 'LLVM\\bin\\clang-format.exe')
                if os.path.exists(win32bin):
                    self.path = win32bin
                    break
        if self.path is None or not os.path.isfile(self.path) or (not self._validate_version()):
            print('ERROR:clang-format not found in $PATH, please install clang-format ' + CLANG_FORMAT_VERSION)
            raise NameError('No suitable clang-format found')
        self.print_lock = threading.Lock()

    def _validate_version(self):
        if False:
            print('Hello World!')
        'Validate clang-format is the expected version\n        '
        cf_version = callo([self.path, '--version'])
        if CLANG_FORMAT_VERSION in cf_version:
            return True
        print('WARNING: clang-format found in path, but incorrect version found at ' + self.path + ' with version: ' + cf_version)
        return False

    def _lint(self, file_name, print_diff):
        if False:
            print('Hello World!')
        'Check the specified file has the correct format\n        '
        with open(file_name, 'rb') as original_text:
            original_file = original_text.read().decode('utf-8')
        formatted_file = callo([self.path, '--style=file', file_name])
        if original_file != formatted_file:
            if print_diff:
                original_lines = original_file.splitlines()
                formatted_lines = formatted_file.splitlines()
                result = difflib.unified_diff(original_lines, formatted_lines)
                with self.print_lock:
                    print('ERROR: Found diff for ' + file_name)
                    print('To fix formatting errors, run %s --style=file -i %s' % (self.path, file_name))
                    for line in result:
                        print(line.rstrip())
            return False
        return True

    def lint(self, file_name):
        if False:
            i = 10
            return i + 15
        'Check the specified file has the correct format\n        '
        return self._lint(file_name, print_diff=True)

    def format(self, file_name):
        if False:
            for i in range(10):
                print('nop')
        'Update the format of the specified file\n        '
        if self._lint(file_name, print_diff=False):
            return True
        formatted = not subprocess.call([self.path, '--style=file', '-i', file_name])
        if sys.platform == 'win32':
            glob_pattern = file_name + '*.TMP'
            for fglob in glob.glob(glob_pattern):
                os.unlink(fglob)
        return formatted

def parallel_process(items, func):
    if False:
        i = 10
        return i + 15
    'Run a set of work items to completion\n    '
    try:
        cpus = cpu_count()
    except NotImplementedError:
        cpus = 1
    task_queue = queue.Queue()
    pp_event = threading.Event()
    pp_result = [True]

    def worker():
        if False:
            i = 10
            return i + 15
        'Worker thread to process work items in parallel\n        '
        while not pp_event.is_set():
            try:
                item = task_queue.get_nowait()
            except queue.Empty:
                pp_event.set()
                return
            try:
                ret = func(item)
            finally:
                task_queue.task_done()
            if not ret:
                print('{} failed on item {}'.format(func, item))
                return
    for item in items:
        task_queue.put(item)
    threads = []
    for cpu in range(cpus):
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    while not pp_event.wait(1) and (not pp_event.is_set()):
        time.sleep(1)
    for thread in threads:
        thread.join()
    return pp_result[0]

def get_base_dir():
    if False:
        for i in range(10):
            print('nop')
    'Get the base directory for mongo repo.\n        This script assumes that it is running in buildscripts/, and uses\n        that to find the base directory.\n    '
    try:
        return subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).rstrip().decode('utf-8')
    except CalledProcessError:
        return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def get_repos():
    if False:
        return 10
    'Get a list of Repos to check clang-format for\n    '
    base_dir = get_base_dir()
    paths = [base_dir]
    return [Repo(p) for p in paths]

class Repo(object):
    """Class encapsulates all knowledge about a git repository, and its metadata
        to run clang-format.
    """

    def __init__(self, path):
        if False:
            i = 10
            return i + 15
        self.path = path
        self.root = self._get_root()

    def _callgito(self, args):
        if False:
            i = 10
            return i + 15
        'Call git for this repository, and return the captured output\n        '
        return callo(['git', '--git-dir', os.path.join(self.path, '.git'), '--work-tree', self.path] + args)

    def _callgit(self, args):
        if False:
            while True:
                i = 10
        'Call git for this repository without capturing output\n        This is designed to be used when git returns non-zero exit codes.\n        '
        return subprocess.call(['git', '--git-dir', os.path.join(self.path, '.git'), '--work-tree', self.path] + args)

    def _get_local_dir(self, path):
        if False:
            print('Hello World!')
        'Get a directory path relative to the git root directory\n        '
        if os.path.isabs(path):
            return os.path.relpath(path, self.root)
        return path

    def get_candidates(self, candidates):
        if False:
            print('Hello World!')
        'Get the set of candidate files to check by querying the repository\n\n        Returns the full path to the file for clang-format to consume.\n        '
        if candidates is not None and len(candidates) > 0:
            candidates = [self._get_local_dir(f) for f in candidates]
            valid_files = list(set(candidates).intersection(self.get_candidate_files()))
        else:
            valid_files = list(self.get_candidate_files())
        valid_files = [os.path.normpath(os.path.join(self.root, f)) for f in valid_files]
        return valid_files

    def get_root(self):
        if False:
            while True:
                i = 10
        'Get the root directory for this repository\n        '
        return self.root

    def _get_root(self):
        if False:
            print('Hello World!')
        'Gets the root directory for this repository from git\n        '
        gito = self._callgito(['rev-parse', '--show-toplevel'])
        return gito.rstrip()

    def _git_ls_files(self, cmd):
        if False:
            print('Hello World!')
        'Run git-ls-files and filter the list of files to a valid candidate list\n        '
        gito = self._callgito(cmd)
        file_list = [line.rstrip() for line in gito.splitlines() if not 'volk' in line]
        file_list = [a for a in file_list if files_match.search(a)]
        return file_list

    def get_candidate_files(self):
        if False:
            print('Hello World!')
        'Query git to get a list of all files in the repo to consider for analysis\n        '
        return self._git_ls_files(['ls-files', '--cached'])

    def get_working_tree_candidate_files(self):
        if False:
            return 10
        'Query git to get a list of all files in the working tree to consider for analysis\n        '
        return self._git_ls_files(['ls-files', '--cached', '--others'])

    def get_working_tree_candidates(self):
        if False:
            print('Hello World!')
        'Get the set of candidate files to check by querying the repository\n\n        Returns the full path to the file for clang-format to consume.\n        '
        valid_files = list(self.get_working_tree_candidate_files())
        valid_files = [os.path.normpath(os.path.join(self.root, f)) for f in valid_files]
        return valid_files

    def is_detached(self):
        if False:
            print('Hello World!')
        'Is the current working tree in a detached HEAD state?\n        '
        return self._callgit(['symbolic-ref', '--quiet', 'HEAD'])

    def is_ancestor(self, parent, child):
        if False:
            return 10
        'Is the specified parent hash an ancestor of child hash?\n        '
        return not self._callgit(['merge-base', '--is-ancestor', parent, child])

    def is_commit(self, sha1):
        if False:
            for i in range(10):
                print('nop')
        'Is the specified hash a valid git commit?\n        '
        return not self._callgit(['cat-file', '-e', '%s^{commit}' % sha1])

    def is_working_tree_dirty(self):
        if False:
            print('Hello World!')
        'Does the current working tree have changes?\n        '
        return self._callgit(['diff', '--quiet'])

    def does_branch_exist(self, branch):
        if False:
            return 10
        'Does the branch exist?\n        '
        return not self._callgit(['rev-parse', '--verify', branch])

    def get_merge_base(self, commit):
        if False:
            print('Hello World!')
        "Get the merge base between 'commit' and HEAD\n        "
        return self._callgito(['merge-base', 'HEAD', commit]).rstrip()

    def get_branch_name(self):
        if False:
            i = 10
            return i + 15
        'Get the current branch name, short form\n           This returns "main", not "refs/head/main"\n           Will not work if the current branch is detached\n        '
        branch = self.rev_parse(['--abbrev-ref', 'HEAD'])
        if branch == 'HEAD':
            raise ValueError('Branch is currently detached')
        return branch

    def add(self, command):
        if False:
            i = 10
            return i + 15
        'git add wrapper\n        '
        return self._callgito(['add'] + command)

    def checkout(self, command):
        if False:
            while True:
                i = 10
        'git checkout wrapper\n        '
        return self._callgito(['checkout'] + command)

    def commit(self, command):
        if False:
            return 10
        'git commit wrapper\n        '
        return self._callgito(['commit'] + command)

    def diff(self, command):
        if False:
            return 10
        'git diff wrapper\n        '
        return self._callgito(['diff'] + command)

    def log(self, command):
        if False:
            print('Hello World!')
        'git log wrapper\n        '
        return self._callgito(['log'] + command)

    def rev_parse(self, command):
        if False:
            while True:
                i = 10
        'git rev-parse wrapper\n        '
        return self._callgito(['rev-parse'] + command).rstrip()

    def rm(self, command):
        if False:
            return 10
        'git rm wrapper\n        '
        return self._callgito(['rm'] + command)

    def show(self, command):
        if False:
            for i in range(10):
                print('nop')
        'git show wrapper\n        '
        return self._callgito(['show'] + command)

def get_list_from_lines(lines):
    if False:
        return 10
    '"Convert a string containing a series of lines into a list of strings\n    '
    return [line.rstrip() for line in lines.splitlines()]

def get_files_to_check_working_tree():
    if False:
        while True:
            i = 10
    'Get a list of files to check form the working tree.\n       This will pick up files not managed by git.\n    '
    repos = get_repos()
    valid_files = list(itertools.chain.from_iterable([r.get_working_tree_candidates() for r in repos]))
    return valid_files

def get_files_to_check():
    if False:
        while True:
            i = 10
    'Get a list of files that need to be checked\n       based on which files are managed by git.\n    '
    repos = get_repos()
    valid_files = list(itertools.chain.from_iterable([r.get_candidates(None) for r in repos]))
    return valid_files

def get_files_to_check_from_patch(patches):
    if False:
        for i in range(10):
            print('nop')
    '\n    Take a patch file generated by git diff,\n    and scan the patch for a list of files to check.\n    '
    candidates = []
    check = re.compile('^diff --git a\\/([a-z\\/\\.\\-_0-9]+) b\\/[a-z\\/\\.\\-_0-9]+')
    candidates = []
    for patch in patches:
        if patch == '-':
            infile = sys.stdin
        else:
            infile = open(patch, 'rb')
        candidates.extend([check.match(line).group(1) for line in infile.readlines() if check.match(line)])
        infile.close()
    repos = get_repos()
    valid_files = list(itertools.chain.from_iterable([r.get_candidates(candidates) for r in repos]))
    return valid_files

def _lint_files(clang_format, files):
    if False:
        while True:
            i = 10
    'Lint a list of files with clang-format\n    '
    try:
        clang_format = ClangFormat(clang_format)
    except NameError as e:
        print(e)
        return False
    lint_clean = parallel_process([os.path.abspath(f) for f in files], clang_format.lint)
    if not lint_clean:
        print('ERROR: Code Style does not match coding style')
        sys.exit(1)

def lint(args):
    if False:
        i = 10
        return i + 15
    'Lint files command entry point\n    '
    if args.patch and args.all:
        print('Only specify patch or all, but not both!')
        return False
    if args.patch:
        files = get_files_to_check_from_patch(args.patch)
    elif args.all:
        files = get_files_to_check_working_tree()
    else:
        files = get_files_to_check()
    if files:
        _lint_files(args.clang_format, files)
    return True

def _format_files(clang_format, files):
    if False:
        for i in range(10):
            print('nop')
    'Format a list of files with clang-format\n    '
    try:
        clang_format = ClangFormat(clang_format)
    except NameError as e:
        print(e)
        return False
    format_clean = parallel_process([os.path.abspath(f) for f in files], clang_format.format)
    if not format_clean:
        print('ERROR: failed to format files')
        sys.exit(1)

def _reformat_branch(clang_format, commit_prior_to_reformat, commit_after_reformat):
    if False:
        i = 10
        return i + 15
    'Reformat a branch made before a clang-format run\n    '
    try:
        clang_format = ClangFormat(clang_format)
    except NameError as e:
        print(e)
        return False
    if os.getcwd() != get_base_dir():
        raise ValueError('reformat-branch must be run from the repo root')
    repo = Repo(get_base_dir())
    if not repo.is_commit(commit_prior_to_reformat):
        raise ValueError("Commit Prior to Reformat '%s' is not a valid commit in this repo" % commit_prior_to_reformat)
    if not repo.is_commit(commit_after_reformat):
        raise ValueError("Commit After Reformat '%s' is not a valid commit in this repo" % commit_after_reformat)
    if not repo.is_ancestor(commit_prior_to_reformat, commit_after_reformat):
        raise ValueError(("Commit Prior to Reformat '%s' is not a valid ancestor of Commit After" + " Reformat '%s' in this repo") % (commit_prior_to_reformat, commit_after_reformat))
    if repo.is_detached():
        raise ValueError('You must not run this script in a detached HEAD state')
    if repo.is_working_tree_dirty():
        raise ValueError('Your working tree has pending changes. You must have a clean working tree before proceeding.')
    merge_base = repo.get_merge_base(commit_prior_to_reformat)
    if not merge_base == commit_prior_to_reformat:
        raise ValueError("Please rebase to '%s' and resolve all conflicts before running this script" % commit_prior_to_reformat)
    merge_base = repo.get_merge_base('main')
    if not merge_base == commit_prior_to_reformat:
        raise ValueError('This branch appears to already have advanced too far through the merge process')
    branch_name = repo.get_branch_name()
    new_branch = '%s-reformatted' % branch_name
    if repo.does_branch_exist(new_branch):
        raise ValueError("The branch '%s' already exists. Please delete the branch '%s', or rename the current branch." % (new_branch, new_branch))
    commits = get_list_from_lines(repo.log(['--reverse', '--pretty=format:%H', '%s..HEAD' % commit_prior_to_reformat]))
    previous_commit_base = commit_after_reformat
    for commit_hash in commits:
        repo.checkout(['--quiet', commit_hash])
        deleted_files = []
        commit_files = get_list_from_lines(repo.diff(['HEAD~', '--name-only']))
        for commit_file in commit_files:
            if not os.path.exists(commit_file):
                print("Skipping file '%s' since it has been deleted in commit '%s'" % (commit_file, commit_hash))
                deleted_files.append(commit_file)
                continue
            if files_match.search(commit_file):
                clang_format.format(commit_file)
            else:
                print("Skipping file '%s' since it is not a file clang_format should format" % commit_file)
        if not repo.is_working_tree_dirty():
            print('Commit %s needed no reformatting' % commit_hash)
        else:
            repo.commit(['--all', '--amend', '--no-edit'])
        previous_commit = repo.rev_parse(['HEAD'])
        repo.checkout(['--quiet', previous_commit_base])
        diff_files = get_list_from_lines(repo.diff(['%s~..%s' % (previous_commit, previous_commit), '--name-only']))
        for diff_file in diff_files:
            if diff_file in deleted_files:
                repo.rm([diff_file])
                continue
            if 'volk' in diff_file:
                continue
            file_contents = repo.show(['%s:%s' % (previous_commit, diff_file)])
            root_dir = os.path.dirname(diff_file)
            if root_dir and (not os.path.exists(root_dir)):
                os.makedirs(root_dir)
            with open(diff_file, 'w+') as new_file:
                new_file.write(file_contents)
            repo.add([diff_file])
        repo.commit(['--reuse-message=%s' % previous_commit])
        previous_commit_base = repo.rev_parse(['HEAD'])
    repo.checkout(['-b', new_branch])
    print('reformat-branch is done running.\n')
    print("A copy of your branch has been made named '%s', and formatted with clang-format.\n" % new_branch)
    print('The original branch has been left unchanged.')
    print("The next step is to rebase the new branch on 'main'.")

def format_func(args):
    if False:
        for i in range(10):
            print('nop')
    'Format files command entry point\n    '
    if args.all and args.branch is not None:
        print('Only specify branch or all, but not both!')
        return False
    if not args.branch:
        if args.all:
            files = get_files_to_check_working_tree()
        else:
            files = get_files_to_check()
        _format_files(args.clang_format, files)
    else:
        _reformat_branch(args.clang_format, *args.branch)

def parse_args():
    if False:
        print('Hello World!')
    '\n    Parse commandline arguments\n    '
    parser = ArgumentParser()
    parser.add_argument('-c', '--clang-format', default='clang-format', help='clang-format binary')
    subparsers = parser.add_subparsers(help='clang-format action', dest='action')
    subparsers.required = True
    lint_parser = subparsers.add_parser('lint', help='Lint-only (no modifications)')
    lint_parser.add_argument('-a', '--all', action='store_true')
    lint_parser.add_argument('-p', '--patch', help='patch to check')
    lint_parser.set_defaults(func=lint)
    format_parser = subparsers.add_parser('format', help='Format files in place')
    format_parser.add_argument('-b', '--branch', nargs=2, default=None, help='specify the commit hash before the format and after the format has been done')
    format_parser.add_argument('-a', '--all', action='store_true')
    format_parser.set_defaults(func=format_func)
    return parser.parse_args()

def main():
    if False:
        i = 10
        return i + 15
    'Main entry point\n    '
    args = parse_args()
    if hasattr(args, 'func'):
        args.func(args)
if __name__ == '__main__':
    main()