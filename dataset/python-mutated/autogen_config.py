"""Generates a configuration options document for Sphinx.

Using this helper tool, a reStructuredText document can be created from
reading the config options from the JupyterQtConsole source code that may
be set in config file, `jupyter_qtconsole_config.py`, and writing to the rST
doc, `config_options.rst`.

"""
import os.path
from qtconsole.qtconsoleapp import JupyterQtConsoleApp
header = 'Configuration options\n=====================\n\nThese options can be set in the configuration file,\n``~/.jupyter/jupyter_qtconsole_config.py``, or\nat the command line when you start Qt console.\n\nYou may enter ``jupyter qtconsole --help-all`` to get information\nabout all available configuration options.\n\nOptions\n-------\n'
destination = os.path.join(os.path.dirname(__file__), 'source/config_options.rst')

def main():
    if False:
        i = 10
        return i + 15
    with open(destination, 'w') as f:
        f.write(header)
        f.write(JupyterQtConsoleApp().document_config_options())
if __name__ == '__main__':
    main()