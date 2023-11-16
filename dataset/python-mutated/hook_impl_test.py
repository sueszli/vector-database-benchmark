from __future__ import annotations
import subprocess
import sys
from unittest import mock
import pytest
import pre_commit.constants as C
from pre_commit import git
from pre_commit.commands import hook_impl
from pre_commit.envcontext import envcontext
from pre_commit.util import cmd_output
from pre_commit.util import make_executable
from testing.fixtures import git_dir
from testing.fixtures import sample_local_config
from testing.fixtures import write_config
from testing.util import cwd
from testing.util import git_commit

def test_validate_config_file_exists(tmpdir):
    if False:
        print('Hello World!')
    cfg = tmpdir.join(C.CONFIG_FILE).ensure()
    hook_impl._validate_config(0, cfg, True)

def test_validate_config_missing(capsys):
    if False:
        i = 10
        return i + 15
    with pytest.raises(SystemExit) as excinfo:
        hook_impl._validate_config(123, 'DNE.yaml', False)
    (ret,) = excinfo.value.args
    assert ret == 1
    assert capsys.readouterr().out == 'No DNE.yaml file was found\n- To temporarily silence this, run `PRE_COMMIT_ALLOW_NO_CONFIG=1 git ...`\n- To permanently silence this, install pre-commit with the --allow-missing-config option\n- To uninstall pre-commit run `pre-commit uninstall`\n'

def test_validate_config_skip_missing_config(capsys):
    if False:
        while True:
            i = 10
    with pytest.raises(SystemExit) as excinfo:
        hook_impl._validate_config(123, 'DNE.yaml', True)
    (ret,) = excinfo.value.args
    assert ret == 123
    expected = '`DNE.yaml` config file not found. Skipping `pre-commit`.\n'
    assert capsys.readouterr().out == expected

def test_validate_config_skip_via_env_variable(capsys):
    if False:
        i = 10
        return i + 15
    with pytest.raises(SystemExit) as excinfo:
        with envcontext((('PRE_COMMIT_ALLOW_NO_CONFIG', '1'),)):
            hook_impl._validate_config(0, 'DNE.yaml', False)
    (ret,) = excinfo.value.args
    assert ret == 0
    expected = '`DNE.yaml` config file not found. Skipping `pre-commit`.\n'
    assert capsys.readouterr().out == expected

def test_run_legacy_does_not_exist(tmpdir):
    if False:
        return 10
    (retv, stdin) = hook_impl._run_legacy('pre-commit', tmpdir, ())
    assert (retv, stdin) == (0, b'')

def test_run_legacy_executes_legacy_script(tmpdir, capfd):
    if False:
        for i in range(10):
            print('nop')
    hook = tmpdir.join('pre-commit.legacy')
    hook.write('#!/usr/bin/env bash\necho hi "$@"\nexit 1\n')
    make_executable(hook)
    (retv, stdin) = hook_impl._run_legacy('pre-commit', tmpdir, ('arg1', 'arg2'))
    assert capfd.readouterr().out.strip() == 'hi arg1 arg2'
    assert (retv, stdin) == (1, b'')

def test_run_legacy_pre_push_returns_stdin(tmpdir):
    if False:
        while True:
            i = 10
    with mock.patch.object(sys.stdin.buffer, 'read', return_value=b'stdin'):
        (retv, stdin) = hook_impl._run_legacy('pre-push', tmpdir, ())
    assert (retv, stdin) == (0, b'stdin')

def test_run_legacy_recursive(tmpdir):
    if False:
        i = 10
        return i + 15
    hook = tmpdir.join('pre-commit.legacy').ensure()
    make_executable(hook)

    def call(*_, **__):
        if False:
            return 10
        return hook_impl._run_legacy('pre-commit', tmpdir, ())
    with mock.patch.object(subprocess, 'run', call):
        with pytest.raises(SystemExit):
            call()

@pytest.mark.parametrize(('hook_type', 'args'), (('pre-commit', []), ('pre-merge-commit', []), ('pre-push', ['branch_name', 'remote_name']), ('commit-msg', ['.git/COMMIT_EDITMSG']), ('post-commit', []), ('post-merge', ['1']), ('pre-rebase', ['main', 'topic']), ('pre-rebase', ['main']), ('post-checkout', ['old_head', 'new_head', '1']), ('post-rewrite', ['amend']), ('prepare-commit-msg', ['.git/COMMIT_EDITMSG']), ('prepare-commit-msg', ['.git/COMMIT_EDITMSG', 'message']), ('prepare-commit-msg', ['.git/COMMIT_EDITMSG', 'commit', 'deadbeef'])))
def test_check_args_length_ok(hook_type, args):
    if False:
        return 10
    hook_impl._check_args_length(hook_type, args)

def test_check_args_length_error_too_many_plural():
    if False:
        while True:
            i = 10
    with pytest.raises(SystemExit) as excinfo:
        hook_impl._check_args_length('pre-commit', ['run', '--all-files'])
    (msg,) = excinfo.value.args
    assert msg == "hook-impl for pre-commit expected 0 arguments but got 2: ['run', '--all-files']"

def test_check_args_length_error_too_many_singular():
    if False:
        i = 10
        return i + 15
    with pytest.raises(SystemExit) as excinfo:
        hook_impl._check_args_length('commit-msg', [])
    (msg,) = excinfo.value.args
    assert msg == 'hook-impl for commit-msg expected 1 argument but got 0: []'

def test_check_args_length_prepare_commit_msg_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(SystemExit) as excinfo:
        hook_impl._check_args_length('prepare-commit-msg', [])
    (msg,) = excinfo.value.args
    assert msg == 'hook-impl for prepare-commit-msg expected 1, 2, or 3 arguments but got 0: []'

def test_check_args_length_pre_rebase_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(SystemExit) as excinfo:
        hook_impl._check_args_length('pre-rebase', [])
    (msg,) = excinfo.value.args
    assert msg == 'hook-impl for pre-rebase expected 1 or 2 arguments but got 0: []'

def test_run_ns_pre_commit():
    if False:
        print('Hello World!')
    ns = hook_impl._run_ns('pre-commit', True, (), b'')
    assert ns is not None
    assert ns.hook_stage == 'pre-commit'
    assert ns.color is True

def test_run_ns_pre_rebase():
    if False:
        return 10
    ns = hook_impl._run_ns('pre-rebase', True, ('main', 'topic'), b'')
    assert ns is not None
    assert ns.hook_stage == 'pre-rebase'
    assert ns.color is True
    assert ns.pre_rebase_upstream == 'main'
    assert ns.pre_rebase_branch == 'topic'
    ns = hook_impl._run_ns('pre-rebase', True, ('main',), b'')
    assert ns is not None
    assert ns.hook_stage == 'pre-rebase'
    assert ns.color is True
    assert ns.pre_rebase_upstream == 'main'
    assert ns.pre_rebase_branch is None

def test_run_ns_commit_msg():
    if False:
        print('Hello World!')
    ns = hook_impl._run_ns('commit-msg', False, ('.git/COMMIT_MSG',), b'')
    assert ns is not None
    assert ns.hook_stage == 'commit-msg'
    assert ns.color is False
    assert ns.commit_msg_filename == '.git/COMMIT_MSG'

def test_run_ns_prepare_commit_msg_one_arg():
    if False:
        for i in range(10):
            print('nop')
    ns = hook_impl._run_ns('prepare-commit-msg', False, ('.git/COMMIT_MSG',), b'')
    assert ns is not None
    assert ns.hook_stage == 'prepare-commit-msg'
    assert ns.color is False
    assert ns.commit_msg_filename == '.git/COMMIT_MSG'

def test_run_ns_prepare_commit_msg_two_arg():
    if False:
        while True:
            i = 10
    ns = hook_impl._run_ns('prepare-commit-msg', False, ('.git/COMMIT_MSG', 'message'), b'')
    assert ns is not None
    assert ns.hook_stage == 'prepare-commit-msg'
    assert ns.color is False
    assert ns.commit_msg_filename == '.git/COMMIT_MSG'
    assert ns.prepare_commit_message_source == 'message'

def test_run_ns_prepare_commit_msg_three_arg():
    if False:
        i = 10
        return i + 15
    ns = hook_impl._run_ns('prepare-commit-msg', False, ('.git/COMMIT_MSG', 'message', 'HEAD'), b'')
    assert ns is not None
    assert ns.hook_stage == 'prepare-commit-msg'
    assert ns.color is False
    assert ns.commit_msg_filename == '.git/COMMIT_MSG'
    assert ns.prepare_commit_message_source == 'message'
    assert ns.commit_object_name == 'HEAD'

def test_run_ns_post_commit():
    if False:
        while True:
            i = 10
    ns = hook_impl._run_ns('post-commit', True, (), b'')
    assert ns is not None
    assert ns.hook_stage == 'post-commit'
    assert ns.color is True

def test_run_ns_post_merge():
    if False:
        for i in range(10):
            print('nop')
    ns = hook_impl._run_ns('post-merge', True, ('1',), b'')
    assert ns is not None
    assert ns.hook_stage == 'post-merge'
    assert ns.color is True
    assert ns.is_squash_merge == '1'

def test_run_ns_post_rewrite():
    if False:
        i = 10
        return i + 15
    ns = hook_impl._run_ns('post-rewrite', True, ('amend',), b'')
    assert ns is not None
    assert ns.hook_stage == 'post-rewrite'
    assert ns.color is True
    assert ns.rewrite_command == 'amend'

def test_run_ns_post_checkout():
    if False:
        return 10
    ns = hook_impl._run_ns('post-checkout', True, ('a', 'b', 'c'), b'')
    assert ns is not None
    assert ns.hook_stage == 'post-checkout'
    assert ns.color is True
    assert ns.from_ref == 'a'
    assert ns.to_ref == 'b'
    assert ns.checkout_type == 'c'

@pytest.fixture
def push_example(tempdir_factory):
    if False:
        while True:
            i = 10
    src = git_dir(tempdir_factory)
    git_commit(cwd=src)
    src_head = git.head_rev(src)
    clone = tempdir_factory.get()
    cmd_output('git', 'clone', src, clone)
    git_commit(cwd=clone)
    clone_head = git.head_rev(clone)
    return (src, src_head, clone, clone_head)

def test_run_ns_pre_push_updating_branch(push_example):
    if False:
        print('Hello World!')
    (src, src_head, clone, clone_head) = push_example
    with cwd(clone):
        args = ('origin', src)
        stdin = f'HEAD {clone_head} refs/heads/b {src_head}\n'.encode()
        ns = hook_impl._run_ns('pre-push', False, args, stdin)
    assert ns is not None
    assert ns.hook_stage == 'pre-push'
    assert ns.color is False
    assert ns.remote_name == 'origin'
    assert ns.remote_url == src
    assert ns.from_ref == src_head
    assert ns.to_ref == clone_head
    assert ns.all_files is False

def test_run_ns_pre_push_new_branch(push_example):
    if False:
        while True:
            i = 10
    (src, src_head, clone, clone_head) = push_example
    with cwd(clone):
        args = ('origin', src)
        stdin = f'HEAD {clone_head} refs/heads/b {hook_impl.Z40}\n'.encode()
        ns = hook_impl._run_ns('pre-push', False, args, stdin)
    assert ns is not None
    assert ns.from_ref == src_head
    assert ns.to_ref == clone_head

def test_run_ns_pre_push_new_branch_existing_rev(push_example):
    if False:
        print('Hello World!')
    (src, src_head, clone, _) = push_example
    with cwd(clone):
        args = ('origin', src)
        stdin = f'HEAD {src_head} refs/heads/b2 {hook_impl.Z40}\n'.encode()
        ns = hook_impl._run_ns('pre-push', False, args, stdin)
    assert ns is None

def test_run_ns_pre_push_ref_with_whitespace(push_example):
    if False:
        print('Hello World!')
    (src, src_head, clone, _) = push_example
    with cwd(clone):
        args = ('origin', src)
        line = f'HEAD^{{/ }} {src_head} refs/heads/b2 {hook_impl.Z40}\n'
        stdin = line.encode()
        ns = hook_impl._run_ns('pre-push', False, args, stdin)
    assert ns is None

def test_pushing_orphan_branch(push_example):
    if False:
        return 10
    (src, src_head, clone, _) = push_example
    cmd_output('git', 'checkout', '--orphan', 'b2', cwd=clone)
    git_commit(cwd=clone, msg='something else to get unique hash')
    clone_rev = git.head_rev(clone)
    with cwd(clone):
        args = ('origin', src)
        stdin = f'HEAD {clone_rev} refs/heads/b2 {hook_impl.Z40}\n'.encode()
        ns = hook_impl._run_ns('pre-push', False, args, stdin)
    assert ns is not None
    assert ns.all_files is True

def test_run_ns_pre_push_deleting_branch(push_example):
    if False:
        print('Hello World!')
    (src, src_head, clone, _) = push_example
    with cwd(clone):
        args = ('origin', src)
        stdin = f'(delete) {hook_impl.Z40} refs/heads/b {src_head}'.encode()
        ns = hook_impl._run_ns('pre-push', False, args, stdin)
    assert ns is None

def test_hook_impl_main_noop_pre_push(cap_out, store, push_example):
    if False:
        for i in range(10):
            print('nop')
    (src, src_head, clone, _) = push_example
    stdin = f'(delete) {hook_impl.Z40} refs/heads/b {src_head}'.encode()
    with mock.patch.object(sys.stdin.buffer, 'read', return_value=stdin):
        with cwd(clone):
            write_config('.', sample_local_config())
            ret = hook_impl.hook_impl(store, config=C.CONFIG_FILE, color=False, hook_type='pre-push', hook_dir='.git/hooks', skip_on_missing_config=False, args=('origin', src))
    assert ret == 0
    assert cap_out.get() == ''

def test_hook_impl_main_runs_hooks(cap_out, tempdir_factory, store):
    if False:
        i = 10
        return i + 15
    with cwd(git_dir(tempdir_factory)):
        write_config('.', sample_local_config())
        ret = hook_impl.hook_impl(store, config=C.CONFIG_FILE, color=False, hook_type='pre-commit', hook_dir='.git/hooks', skip_on_missing_config=False, args=())
    assert ret == 0
    expected = 'Block if "DO NOT COMMIT" is found....................(no files to check)Skipped\n'
    assert cap_out.get() == expected