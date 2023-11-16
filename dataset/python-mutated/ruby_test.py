from __future__ import annotations
import tarfile
from unittest import mock
import pytest
import pre_commit.constants as C
from pre_commit import parse_shebang
from pre_commit.envcontext import envcontext
from pre_commit.languages import ruby
from pre_commit.languages.ruby import _resource_bytesio
from pre_commit.store import _make_local_repo
from testing.language_helpers import run_language
from testing.util import cwd
from testing.util import xfailif_windows
ACTUAL_GET_DEFAULT_VERSION = ruby.get_default_version.__wrapped__

@pytest.fixture
def find_exe_mck():
    if False:
        return 10
    with mock.patch.object(parse_shebang, 'find_executable') as mck:
        yield mck

def test_uses_default_version_when_not_available(find_exe_mck):
    if False:
        while True:
            i = 10
    find_exe_mck.return_value = None
    assert ACTUAL_GET_DEFAULT_VERSION() == C.DEFAULT

def test_uses_system_if_both_gem_and_ruby_are_available(find_exe_mck):
    if False:
        i = 10
        return i + 15
    find_exe_mck.return_value = '/path/to/exe'
    assert ACTUAL_GET_DEFAULT_VERSION() == 'system'

@pytest.mark.parametrize('filename', ('rbenv.tar.gz', 'ruby-build.tar.gz', 'ruby-download.tar.gz'))
def test_archive_root_stat(filename):
    if False:
        while True:
            i = 10
    with _resource_bytesio(filename) as f:
        with tarfile.open(fileobj=f) as tarf:
            (root, _, _) = filename.partition('.')
            assert oct(tarf.getmember(root).mode) == '0o755'

def _setup_hello_world(tmp_path):
    if False:
        while True:
            i = 10
    bin_dir = tmp_path.joinpath('bin')
    bin_dir.mkdir()
    bin_dir.joinpath('ruby_hook').write_text("#!/usr/bin/env ruby\nputs 'Hello world from a ruby hook'\n")
    gemspec = "Gem::Specification.new do |s|\n    s.name = 'ruby_hook'\n    s.version = '0.1.0'\n    s.authors = ['Anthony Sottile']\n    s.summary = 'A ruby hook!'\n    s.description = 'A ruby hook!'\n    s.files = ['bin/ruby_hook']\n    s.executables = ['ruby_hook']\nend\n"
    tmp_path.joinpath('ruby_hook.gemspec').write_text(gemspec)

def test_ruby_hook_system(tmp_path):
    if False:
        while True:
            i = 10
    assert ruby.get_default_version() == 'system'
    _setup_hello_world(tmp_path)
    ret = run_language(tmp_path, ruby, 'ruby_hook')
    assert ret == (0, b'Hello world from a ruby hook\n')

def test_ruby_with_user_install_set(tmp_path):
    if False:
        while True:
            i = 10
    gemrc = tmp_path.joinpath('gemrc')
    gemrc.write_text('gem: --user-install\n')
    with envcontext((('GEMRC', str(gemrc)),)):
        test_ruby_hook_system(tmp_path)

def test_ruby_additional_deps(tmp_path):
    if False:
        i = 10
        return i + 15
    _make_local_repo(tmp_path)
    ret = run_language(tmp_path, ruby, 'ruby -e', args=('require "tins"',), deps=('tins',))
    assert ret == (0, b'')

@xfailif_windows
def test_ruby_hook_default(tmp_path):
    if False:
        print('Hello World!')
    _setup_hello_world(tmp_path)
    (out, ret) = run_language(tmp_path, ruby, 'rbenv --help', version='default')
    assert out == 0
    assert ret.startswith(b'Usage: rbenv ')

@xfailif_windows
def test_ruby_hook_language_version(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    _setup_hello_world(tmp_path)
    tmp_path.joinpath('bin', 'ruby_hook').write_text("#!/usr/bin/env ruby\nputs RUBY_VERSION\nputs 'Hello world from a ruby hook'\n")
    ret = run_language(tmp_path, ruby, 'ruby_hook', version='3.2.0')
    assert ret == (0, b'3.2.0\nHello world from a ruby hook\n')

@xfailif_windows
def test_ruby_with_bundle_disable_shared_gems(tmp_path):
    if False:
        while True:
            i = 10
    workdir = tmp_path.joinpath('workdir')
    workdir.mkdir()
    workdir.joinpath('Gemfile').write_text('source ""\ngem "lol_hai"\n')
    bundle = workdir.joinpath('.bundle')
    bundle.mkdir()
    bundle.joinpath('config').write_text('BUNDLE_DISABLE_SHARED_GEMS: true\nBUNDLE_PATH: vendor/gem\n')
    with cwd(workdir):
        test_ruby_hook_language_version(tmp_path)