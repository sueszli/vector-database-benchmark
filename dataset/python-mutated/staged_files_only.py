from __future__ import annotations
import contextlib
import logging
import os.path
import time
from collections.abc import Generator
from pre_commit import git
from pre_commit.errors import FatalError
from pre_commit.util import CalledProcessError
from pre_commit.util import cmd_output
from pre_commit.util import cmd_output_b
from pre_commit.xargs import xargs
logger = logging.getLogger('pre_commit')
_CHECKOUT_CMD = ('git', '-c', 'submodule.recurse=0', 'checkout', '--', '.')

def _git_apply(patch: str) -> None:
    if False:
        print('Hello World!')
    args = ('apply', '--whitespace=nowarn', patch)
    try:
        cmd_output_b('git', *args)
    except CalledProcessError:
        cmd_output_b('git', '-c', 'core.autocrlf=false', *args)

@contextlib.contextmanager
def _intent_to_add_cleared() -> Generator[None, None, None]:
    if False:
        print('Hello World!')
    intent_to_add = git.intent_to_add_files()
    if intent_to_add:
        logger.warning('Unstaged intent-to-add files detected.')
        xargs(('git', 'rm', '--cached', '--'), intent_to_add)
        try:
            yield
        finally:
            xargs(('git', 'add', '--intent-to-add', '--'), intent_to_add)
    else:
        yield

@contextlib.contextmanager
def _unstaged_changes_cleared(patch_dir: str) -> Generator[None, None, None]:
    if False:
        i = 10
        return i + 15
    tree = cmd_output('git', 'write-tree')[1].strip()
    diff_cmd = ('git', 'diff-index', '--ignore-submodules', '--binary', '--exit-code', '--no-color', '--no-ext-diff', tree, '--')
    (retcode, diff_stdout, diff_stderr) = cmd_output_b(*diff_cmd, check=False)
    if retcode == 0:
        yield
    elif retcode == 1 and diff_stdout.strip():
        patch_filename = f'patch{int(time.time())}-{os.getpid()}'
        patch_filename = os.path.join(patch_dir, patch_filename)
        logger.warning('Unstaged files detected.')
        logger.info(f'Stashing unstaged files to {patch_filename}.')
        os.makedirs(patch_dir, exist_ok=True)
        with open(patch_filename, 'wb') as patch_file:
            patch_file.write(diff_stdout)
        no_checkout_env = dict(os.environ, _PRE_COMMIT_SKIP_POST_CHECKOUT='1')
        try:
            cmd_output_b(*_CHECKOUT_CMD, env=no_checkout_env)
            yield
        finally:
            try:
                _git_apply(patch_filename)
            except CalledProcessError:
                logger.warning('Stashed changes conflicted with hook auto-fixes... Rolling back fixes...')
                cmd_output_b(*_CHECKOUT_CMD, env=no_checkout_env)
                _git_apply(patch_filename)
            logger.info(f'Restored changes from {patch_filename}.')
    else:
        e = CalledProcessError(retcode, diff_cmd, b'', diff_stderr)
        raise FatalError(f'pre-commit failed to diff -- perhaps due to permissions?\n\n{e}')

@contextlib.contextmanager
def staged_files_only(patch_dir: str) -> Generator[None, None, None]:
    if False:
        print('Hello World!')
    'Clear any unstaged changes from the git working directory inside this\n    context.\n    '
    with _intent_to_add_cleared(), _unstaged_changes_cleared(patch_dir):
        yield