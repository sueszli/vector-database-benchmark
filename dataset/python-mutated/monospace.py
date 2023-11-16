"""A Jupyter magic to choose a real monospaced fonts, if available."""
from IPython.display import HTML, display
from IPython.core.magic import line_magic, Magics, magics_class

@magics_class
class MonospacedOutput(Magics):
    """A class for setting "Courier New" for output code."""

    @line_magic
    def monospaced_output(self, line='', cell=None):
        if False:
            print('Hello World!')
        'A Jupyter magic function to set "Courier New" for output code.'
        html = "<style type='text/css'>\n        code, kbd, pre, samp {font-family: Courier New,monospace;line-height: 1.1;}</style>"
        display(HTML(html))