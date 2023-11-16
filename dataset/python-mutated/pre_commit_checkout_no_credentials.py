from __future__ import annotations
import sys
from pathlib import Path
import yaml
from rich.console import Console
if __name__ not in ('__main__', '__mp_main__'):
    raise SystemExit(f'This file is intended to be executed as an executable program. You cannot use it as a module.To run this script, run the ./{__file__} command [FILE] ...')
console = Console(color_system='standard', width=200)

def check_file(the_file: Path) -> int:
    if False:
        while True:
            i = 10
    'Returns number of wrong checkout instructions in the workflow file'
    error_num = 0
    res = yaml.safe_load(the_file.read_text())
    console.print(f'Checking file [yellow]{the_file}[/]')
    for job in res['jobs'].values():
        for step in job['steps']:
            uses = step.get('uses')
            pretty_step = yaml.safe_dump(step, indent=2)
            if uses is not None and uses.startswith('actions/checkout'):
                with_clause = step.get('with')
                if with_clause is None:
                    console.print(f'\n[red]The `with` clause is missing in step:[/]\n\n{pretty_step}')
                    error_num += 1
                    continue
                path = with_clause.get('path')
                if path == 'constraints':
                    continue
                persist_credentials = with_clause.get('persist-credentials')
                if persist_credentials is None:
                    console.print(f'\n[red]The `with` clause does not have persist-credentials in step:[/]\n\n{pretty_step}')
                    error_num += 1
                    continue
                elif persist_credentials:
                    console.print(f'\n[red]The `with` clause have persist-credentials=True in step:[/]\n\n{pretty_step}')
                    error_num += 1
                    continue
    return error_num
if __name__ == '__main__':
    total_err_num = 0
    for a_file in sys.argv[1:]:
        total_err_num += check_file(Path(a_file))
    if total_err_num:
        console.print('\n[red]There are some checkout instructions in github workflows that have no "persist_credentials"\nset to False.[/]\n\nFor security reasons - make sure all of the checkout actions have persist_credentials set, similar to:\n\n  - name: "Checkout ${{ github.ref }} ( ${{ github.sha }} )"\n    uses: actions/checkout@v4\n    with:\n      persist-credentials: false\n\n')
        sys.exit(1)