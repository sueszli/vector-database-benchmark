"""
This module implements Figlet text renderer.
"""
from pyfiglet import Figlet, DEFAULT_FONT
from asciimatics.renderers.base import StaticRenderer

class FigletText(StaticRenderer):
    """
    This class renders the supplied text using the specified Figlet font.
    See http://www.figlet.org/ for details of available fonts.
    """

    def __init__(self, text, font=DEFAULT_FONT, width=200):
        if False:
            print('Hello World!')
        '\n        :param text: The text string to convert with Figlet.\n        :param font: The Figlet font to use (optional).\n        :param width: The maximum width for this text in characters.\n        '
        super().__init__()
        self._images = [Figlet(font=font, width=width).renderText(text)]