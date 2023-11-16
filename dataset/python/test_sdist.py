"""
Verify the contents of our sdists
"""

import tarfile
from fnmatch import fnmatch
from pathlib import Path
from subprocess import run

import pytest

repo = Path(__file__).parent.parent.resolve()


@pytest.fixture
def sdist():
    path = list(repo.glob("dist/*.tar.gz"))[0]
    return tarfile.open(path)


@pytest.fixture
def sdist_files(sdist):
    paths = set()
    for name in sdist.getnames():
        # 'pyzmq-25.1.0.dev0/zmq/utils/config.json'
        root, _, relative = name.partition("/")
        paths.add(relative)
    return paths


@pytest.fixture
def git_files():
    p = run(["git", "ls-files"], cwd=repo, capture_output=True, text=True)
    paths = set()
    for line in p.stdout.splitlines():
        paths.add(line)
    return paths


def test_git_files(sdist_files, git_files):
    missing_git_files = git_files.difference(sdist_files)
    assert missing_git_files == set()


@pytest.mark.parametrize(
    "path",
    [
        # bundled zeromq
        "bundled/zeromq/COPYING",
        "bundled/zeromq/COPYING.LESSER",
        "bundled/zeromq/include/zmq.h",
        "bundled/zeromq/src/zmq.cpp",
        "bundled/zeromq/external/wepoll/license.txt",
        "bundled/zeromq/external/wepoll/wepoll.h",
        # Cython-generated files
        "zmq/backend/cython/_zmq.c",
    ],
)
def test_included(sdist_files, path):
    assert path in sdist_files


@pytest.mark.parametrize(
    "path",
    [
        "bundled/zeromq/src/platform.hpp",
        "**/*.so",
        "**/__pycache__",
    ],
)
def test_excluded(sdist_files, path):
    matches = [f for f in sdist_files if fnmatch(f, path)]
    assert not matches
