import subprocess
import sys
import click

def main():
    if False:
        for i in range(10):
            print('nop')
    standard_cli = ['streamlit', 'test', 'prog_name']
    if not _can_run_streamlit(standard_cli):
        sys.exit('Failed to run `streamlit test prog_name`')
    module_cli = ['python', '-m', 'streamlit', 'test', 'prog_name']
    if not _can_run_streamlit(module_cli):
        sys.exit('Failed to run `python -m streamlit test prog_name`')
    unsupported_module_cli = ['python', '-m', 'streamlit.cli', 'test', 'prog_name']
    if _can_run_streamlit(unsupported_module_cli):
        sys.exit('`python -m streamlit.cli test prog_name` should not run')
    click.secho('CLI smoke tests succeeded!', fg='green', bold=True)

def _can_run_streamlit(command_list):
    if False:
        i = 10
        return i + 15
    result = subprocess.run(command_list, stdout=subprocess.DEVNULL)
    return result.returncode == 0
if __name__ == '__main__':
    main()