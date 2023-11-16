from __future__ import annotations
import re_assert
from pre_commit.languages import dart
from pre_commit.store import _make_local_repo
from testing.language_helpers import run_language

def test_dart(tmp_path):
    if False:
        print('Hello World!')
    pubspec_yaml = "environment:\n  sdk: '>=2.10.0 <3.0.0'\n\nname: hello_world_dart\n\nexecutables:\n    hello-world-dart:\n\ndependencies:\n  ansicolor: ^2.0.1\n"
    hello_world_dart_dart = 'import \'package:ansicolor/ansicolor.dart\';\n\nvoid main() {\n    AnsiPen pen = new AnsiPen()..red();\n    print("hello hello " + pen("world"));\n}\n'
    tmp_path.joinpath('pubspec.yaml').write_text(pubspec_yaml)
    bin_dir = tmp_path.joinpath('bin')
    bin_dir.mkdir()
    bin_dir.joinpath('hello-world-dart.dart').write_text(hello_world_dart_dart)
    expected = (0, b'hello hello world\n')
    assert run_language(tmp_path, dart, 'hello-world-dart') == expected

def test_dart_additional_deps(tmp_path):
    if False:
        i = 10
        return i + 15
    _make_local_repo(str(tmp_path))
    ret = run_language(tmp_path, dart, 'hello-world-dart', deps=('hello_world_dart',))
    assert ret == (0, b'hello hello world\n')

def test_dart_additional_deps_versioned(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    _make_local_repo(str(tmp_path))
    (ret, out) = run_language(tmp_path, dart, 'secure-random -l 4 -b 16', deps=('encrypt:5.0.0',))
    assert ret == 0
    re_assert.Matches('^[a-f0-9]{8}\n$').assert_matches(out.decode())