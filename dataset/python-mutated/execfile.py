import pathlib
import sys
import click

@click.command(name='execfile', context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument('filename', required=True)
def execfile(filename):
    if False:
        i = 10
        return i + 15
    'Execute a script.\n\n    This is very similar to `exec`, with the following differences:\n\n    - The following header is implicitly executed before the script, regardless\n      of whether the script itself does something similar:\n\n         from sentry.runner import configure; configure()\n         from django.conf import settings\n         from sentry.models import *\n\n    - No support for the -c option.\n\n    - Exceptions are not wrapped, line numbers match in any reported exception and the\n      script.\n\n    - __file__ is set to the filename of the script.\n    '
    filename = pathlib.Path(filename)
    preamble = '\n'.join(['from sentry.runner import configure; configure()', 'from django.conf import settings', 'from sentry.models import *'])
    script_globals = {'__name__': '__main__', '__file__': str(filename)}
    preamble_code = compile(preamble, filename, 'exec')
    exec(preamble_code, script_globals, script_globals)
    sys.argv = sys.argv[1:]
    script_code = compile(filename.read_text(), filename, 'exec')
    exec(script_code, script_globals, script_globals)