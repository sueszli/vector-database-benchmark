from __future__ import annotations
import re
import sys
from pathlib import Path
from rich.console import Console
if __name__ not in ('__main__', '__mp_main__'):
    raise SystemExit(f'This file is intended to be executed as an executable program. You cannot use it as a module.To execute this script, run ./{__file__} [FILE] ...')
console = Console(color_system='standard', width=200)
errors: list[str] = []
WATCHER_APPEND_INSTRUCTION = 'list(dag.tasks) >> watcher()'
PYTEST_FUNCTION = '\nfrom tests.system.utils import get_test_run  # noqa: E402\n\n# Needed to run the example DAG with pytest (see: tests/system/README.md#run_via_pytest)\ntest_run = get_test_run(dag)\n'
PYTEST_FUNCTION_PATTERN = re.compile('from tests\\.system\\.utils import get_test_run(?:  # noqa: E402)?\\s+(?:# .+\\))?\\s+test_run = get_test_run\\(dag\\)')

def _check_file(file: Path):
    if False:
        while True:
            i = 10
    content = file.read_text()
    if 'from tests.system.utils.watcher import watcher' in content:
        index = content.find(WATCHER_APPEND_INSTRUCTION)
        if index == -1:
            errors.append(f'[red]The example {file} imports tests.system.utils.watcher but does not use it properly![/]\n\n[yellow]Make sure you have:[/]\n\n        {WATCHER_APPEND_INSTRUCTION}\n\n[yellow]as the last instruction in your example DAG.[/]\n')
        else:
            operator_leftshift_index = content.find('<<', index + len(WATCHER_APPEND_INSTRUCTION))
            operator_rightshift_index = content.find('>>', index + len(WATCHER_APPEND_INSTRUCTION))
            if operator_leftshift_index != -1 or operator_rightshift_index != -1:
                errors.append(f'[red]In the example {file} watcher is not the last instruction in your DAG (there are << or >> operators after it)![/]\n\n[yellow]Make sure you have:[/]\n        {WATCHER_APPEND_INSTRUCTION}\n\n[yellow]as the last instruction in your example DAG.[/]\n')
    if not PYTEST_FUNCTION_PATTERN.search(content):
        errors.append(f'[yellow]The example {file} missed the pytest function at the end.[/]\n\nAll example tests should have this function added:\n\n' + PYTEST_FUNCTION + '\n\n[yellow]Automatically adding it now![/]\n')
        file.write_text(content + '\n' + PYTEST_FUNCTION)
if __name__ == '__main__':
    for file in sys.argv[1:]:
        _check_file(Path(file))
    if errors:
        console.print('[red]There were some errors in the example files[/]\n')
        for error in errors:
            console.print(error)
        sys.exit(1)