from __future__ import annotations
import pytest
from pre_commit.errors import FatalError
from pre_commit.languages import haskell
from pre_commit.util import win_exe
from testing.language_helpers import run_language

def test_run_example_executable(tmp_path):
    if False:
        print('Hello World!')
    example_cabal = 'cabal-version:      2.4\nname:               example\nversion:            0.1.0.0\n\nexecutable example\n    main-is:          Main.hs\n\n    build-depends:    base >=4\n    default-language: Haskell2010\n'
    main_hs = 'module Main where\n\nmain :: IO ()\nmain = putStrLn "Hello, Haskell!"\n'
    tmp_path.joinpath('example.cabal').write_text(example_cabal)
    tmp_path.joinpath('Main.hs').write_text(main_hs)
    result = run_language(tmp_path, haskell, 'example')
    assert result == (0, b'Hello, Haskell!\n')
    exe = tmp_path.joinpath(win_exe('hs_env-default/bin/example'))
    assert exe.is_file()
    assert not exe.is_symlink()

def test_run_dep(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    result = run_language(tmp_path, haskell, 'hello', deps=['hello'])
    assert result == (0, b'Hello, World!\n')

def test_run_empty(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(FatalError) as excinfo:
        run_language(tmp_path, haskell, 'example')
    (msg,) = excinfo.value.args
    assert msg == 'Expected .cabal files or additional_dependencies'