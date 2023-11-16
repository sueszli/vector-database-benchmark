from __future__ import annotations
from pre_commit.commands.sample_config import sample_config

def test_sample_config(capsys):
    if False:
        while True:
            i = 10
    ret = sample_config()
    assert ret == 0
    (out, _) = capsys.readouterr()
    assert out == '# See https://pre-commit.com for more information\n# See https://pre-commit.com/hooks.html for more hooks\nrepos:\n-   repo: https://github.com/pre-commit/pre-commit-hooks\n    rev: v3.2.0\n    hooks:\n    -   id: trailing-whitespace\n    -   id: end-of-file-fixer\n    -   id: check-yaml\n    -   id: check-added-large-files\n'