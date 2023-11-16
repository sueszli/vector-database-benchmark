"""Quick code snippets for embedding IPython into other programs.

See embed_class_long.py for full details, this file has the bare minimum code for
cut and paste use once you understand how to use the system."""
try:
    get_ipython
except NameError:
    banner = exit_msg = ''
else:
    banner = '*** Nested interpreter ***'
    exit_msg = '*** Back in main IPython ***'
from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed(banner1=banner, exit_msg=exit_msg)
from IPython import embed
try:
    get_ipython
except NameError:
    from IPython.terminal.embed import InteractiveShellEmbed
    ipshell = InteractiveShellEmbed()
else:

    def ipshell():
        if False:
            while True:
                i = 10
        pass