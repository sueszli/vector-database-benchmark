from __future__ import annotations
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING
from airflow_breeze.utils.console import MessageType, get_console
from airflow_breeze.utils.path_utils import skip_group_output
if TYPE_CHECKING:
    from airflow_breeze.utils.parallel import Output
_in_ci_group = False

@contextmanager
def ci_group(title: str, message_type: MessageType | None=MessageType.INFO, output: Output | None=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    If used in GitHub Action, creates an expandable group in the GitHub Action log.\n    Otherwise, display simple text groups.\n\n    For more information, see:\n    https://docs.github.com/en/free-pro-team@latest/actions/reference/workflow-commands-for-github-actions#grouping-log-lines\n    '
    global _in_ci_group
    if _in_ci_group or skip_group_output():
        yield
        return
    if os.environ.get('GITHUB_ACTIONS', 'false') != 'true':
        if message_type is not None:
            get_console(output=output).print(f'\n[{message_type.value}]{title}\n')
        else:
            get_console(output=output).print(f'\n{title}\n')
        yield
        return
    _in_ci_group = True
    if message_type is not None:
        get_console().print(f'::group::[{message_type.value}]{title}[/]')
    else:
        get_console().print(f'::group::{title}')
    try:
        yield
    finally:
        get_console().print('::endgroup::')
        _in_ci_group = False