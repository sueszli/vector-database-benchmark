'''Welcome to the Salt repl which exposes the execution environment of a minion in
a pre-configured Python shell

__opts__, __salt__, __grains__, and __pillar__ are available.

Jinja can be tested with full access to the above structures in the usual way:

    JINJA("""\\
        I am {{ salt['cmd.run']('whoami') }}.

        {% if otherstuff %}
        Some other stuff here
        {% endif %}
    """, otherstuff=True)

A history file is maintained in ~/.saltsh_history.
completion behavior can be customized via the ~/.inputrc file.

'''
import atexit
import builtins
import os
import pprint
import readline
import sys
from code import InteractiveConsole
import jinja2
import salt.client
import salt.config
import salt.loader
import salt.output
import salt.pillar
import salt.runner
import salt.utils.yaml
HISTFILE = '{HOME}/.saltsh_history'.format(**os.environ)

def savehist():
    if False:
        i = 10
        return i + 15
    '\n    Save the history file\n    '
    readline.write_history_file(HISTFILE)

def get_salt_vars():
    if False:
        i = 10
        return i + 15
    '\n    Return all the Salt-usual double-under data structures for a minion\n    '
    __opts__ = salt.config.client_config(os.environ.get('SALT_MINION_CONFIG', '/etc/salt/minion'))
    if 'grains' not in __opts__ or not __opts__['grains']:
        __opts__['grains'] = salt.loader.grains(__opts__)
    if 'file_client' not in __opts__ or not __opts__['file_client']:
        __opts__['file_client'] = 'local'
    if 'id' not in __opts__ or not __opts__['id']:
        __opts__['id'] = 'saltsh_mid'
    __salt__ = salt.loader.minion_mods(__opts__)
    __grains__ = __opts__['grains']
    if __opts__['file_client'] == 'local':
        __pillar__ = salt.pillar.get_pillar(__opts__, __grains__, __opts__.get('id'), __opts__.get('saltenv')).compile_pillar()
    else:
        __pillar__ = {}
    JINJA = lambda x, **y: jinja2.Template(x).render(grains=__grains__, salt=__salt__, opts=__opts__, pillar=__pillar__, **y)
    return locals()

def main():
    if False:
        while True:
            i = 10
    '\n    The main entry point\n    '
    salt_vars = get_salt_vars()

    def salt_outputter(value):
        if False:
            print('Hello World!')
        "\n        Use Salt's outputters to print values to the shell\n        "
        if value is not None:
            builtins._ = value
            salt.output.display_output(value, '', salt_vars['__opts__'])
    sys.displayhook = salt_outputter
    readline.set_history_length(300)
    if os.path.exists(HISTFILE):
        readline.read_history_file(HISTFILE)
    atexit.register(savehist)
    atexit.register(lambda : sys.stdout.write('Salt you later!\n'))
    saltrepl = InteractiveConsole(locals=salt_vars)
    saltrepl.interact(banner=__doc__)
if __name__ == '__main__':
    main()