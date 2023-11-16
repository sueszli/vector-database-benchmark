from __future__ import annotations
from pre_commit.languages import perl
from pre_commit.store import _make_local_repo
from pre_commit.util import make_executable
from testing.language_helpers import run_language

def test_perl_install(tmp_path):
    if False:
        i = 10
        return i + 15
    makefile_pl = 'use strict;\nuse warnings;\n\nuse ExtUtils::MakeMaker;\n\nWriteMakefile(\n    NAME => "PreCommitHello",\n    VERSION_FROM => "lib/PreCommitHello.pm",\n    EXE_FILES => [qw(bin/pre-commit-perl-hello)],\n);\n'
    bin_perl_hello = '#!/usr/bin/env perl\n\nuse strict;\nuse warnings;\nuse PreCommitHello;\n\nPreCommitHello::hello();\n'
    lib_hello_pm = 'package PreCommitHello;\n\nuse strict;\nuse warnings;\n\nour $VERSION = "0.1.0";\n\nsub hello {\n    print "Hello from perl-commit Perl!\n";\n}\n\n1;\n'
    tmp_path.joinpath('Makefile.PL').write_text(makefile_pl)
    bin_dir = tmp_path.joinpath('bin')
    bin_dir.mkdir()
    exe = bin_dir.joinpath('pre-commit-perl-hello')
    exe.write_text(bin_perl_hello)
    make_executable(exe)
    lib_dir = tmp_path.joinpath('lib')
    lib_dir.mkdir()
    lib_dir.joinpath('PreCommitHello.pm').write_text(lib_hello_pm)
    ret = run_language(tmp_path, perl, 'pre-commit-perl-hello')
    assert ret == (0, b'Hello from perl-commit Perl!\n')

def test_perl_additional_dependencies(tmp_path):
    if False:
        return 10
    _make_local_repo(str(tmp_path))
    (ret, out) = run_language(tmp_path, perl, 'perltidy --version', deps=('SHANCOCK/Perl-Tidy-20211029.tar.gz',))
    assert ret == 0
    assert out.startswith(b'This is perltidy, v20211029')