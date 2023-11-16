"""Runs all the scripts in the examples folder (except this one)."""
import os
import click
auto_run = False
EXAMPLE_DIR = 'examples'
EXCLUDED_FILENAMES = ('mnist-cnn.py', 'caching.py')

def run_commands(section_header, commands, skip_last_input=False, comment=None):
    if False:
        i = 10
        return i + 15
    'Run a list of commands, displaying them within the given section.'
    global auto_run
    for (i, command) in enumerate(commands):
        vars = {'section_header': section_header, 'total': len(commands), 'command': command, 'v': i + 1}
        click.secho('\nRunning %(section_header)s %(v)s/%(total)s : %(command)s' % vars, bold=True)
        click.secho('\n%(v)s/%(total)s : %(command)s' % vars, fg='yellow', bold=True)
        if comment:
            click.secho(comment)
        os.system(command)
        last_command = i + 1 == len(commands)
        if not (auto_run or (last_command and skip_last_input)):
            click.secho('Press [enter] to continue or [a] to continue on auto:\n> ', nl=False)
            response = click.getchar()
            if response == 'a':
                print('Turning on auto run.')
                auto_run = True

def main():
    if False:
        while True:
            i = 10
    run_commands('Basic Commands', ['streamlit version'])
    run_commands('Standard System Errors', ['streamlit run does_not_exist.py'], comment='Checks to see that file not found error is caught')
    run_commands('Hello script', ['streamlit hello'])
    run_commands('Examples', ['streamlit run %(EXAMPLE_DIR)s/%(filename)s' % {'EXAMPLE_DIR': EXAMPLE_DIR, 'filename': filename} for filename in os.listdir(EXAMPLE_DIR) if filename.endswith('.py') and filename not in EXCLUDED_FILENAMES])
    run_commands('Caching', ['streamlit cache clear', 'streamlit run %s/caching.py' % EXAMPLE_DIR])
    run_commands('MNIST', ['streamlit run %s/mnist-cnn.py' % EXAMPLE_DIR], skip_last_input=True)
    click.secho('\n\nCompleted all tests!', bold=True)
if __name__ == '__main__':
    main()