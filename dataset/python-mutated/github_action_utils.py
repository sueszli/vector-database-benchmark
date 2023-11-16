from __future__ import annotations
import os
from contextlib import contextmanager

@contextmanager
def with_group(title):
    if False:
        for i in range(10):
            print('nop')
    '\n    If used in GitHub Action, creates an expandable group in the GitHub Action log.\n    Otherwise, display simple text groups.\n\n    For more information, see:\n    https://docs.github.com/en/free-pro-team@latest/actions/reference/workflow-commands-for-github-actions#grouping-log-lines\n    '
    if os.environ.get('GITHUB_ACTIONS', 'false') != 'true':
        print('#' * 20, title, '#' * 20)
        yield
        return
    print(f'::group::{title}')
    print()
    yield
    print('\x1b[0m')
    print('::endgroup::')