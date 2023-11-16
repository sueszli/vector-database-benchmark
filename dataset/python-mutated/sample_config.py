from __future__ import annotations
SAMPLE_CONFIG = '# See https://pre-commit.com for more information\n# See https://pre-commit.com/hooks.html for more hooks\nrepos:\n-   repo: https://github.com/pre-commit/pre-commit-hooks\n    rev: v3.2.0\n    hooks:\n    -   id: trailing-whitespace\n    -   id: end-of-file-fixer\n    -   id: check-yaml\n    -   id: check-added-large-files\n'

def sample_config() -> int:
    if False:
        for i in range(10):
            print('nop')
    print(SAMPLE_CONFIG, end='')
    return 0