"""
This module provides `Sprites` to create animation effects with Paths.  For more details see
http://asciimatics.readthedocs.io/en/latest/animation.html
"""
import random
from asciimatics.effects import Sprite
from asciimatics.renderers import StaticRenderer
from asciimatics.screen import Screen
sam_default = ["\n    ______\n  .`      `.\n /   -  -   \\\n|     __     |\n|            |\n \\          /\n  '.______.'\n", "\n    ______\n  .`      `.\n /   o  o   \\\n|     __     |\n|            |\n \\          /\n  '.______.'\n"]
sam_left = "\n    ______\n  .`      `.\n / o        \\\n|            |\n|--          |\n \\          /\n  '.______.'\n"
sam_right = "\n    ______\n  .`      `.\n /        o \\\n|            |\n|          --|\n \\          /\n  '.______.'\n"
sam_down = "\n    ______\n  .`      `.\n /          \\\n|            |\n|    ^  ^    |\n \\   __     /\n  '.______.'\n"
sam_up = "\n    ______\n  .`  __  `.\n /   v  v   \\\n|            |\n|            |\n \\          /\n  '.______.'\n"
left_arrow = '\n /____\n/\n\\ ____\n \\\n'
up_arrow = '\n  /\\\n /  \\\n/|  |\\\n |  |\n '
right_arrow = '\n____\\\n     \\\n____ /\n    /\n'
down_arrow = '\n |  |\n\\|  |/\n \\  /\n  \\/\n '
default_arrow = ['\n  /\\\n /  \\\n/|><|\\\n |  |\n ', '\n  /\\\n /  \\\n/|oo|\\\n |  |\n ']

def _blink():
    if False:
        return 10
    if random.random() > 0.9:
        return 0
    else:
        return 1

class Sam(Sprite):
    """
    Sam Paul sprite - an simple sample animated character.
    """

    def __init__(self, screen, path, start_frame=0, stop_frame=0):
        if False:
            print('Hello World!')
        '\n        See :py:obj:`.Sprite` for details.\n        '
        super().__init__(screen, renderer_dict={'default': StaticRenderer(images=sam_default, animation=_blink), 'left': StaticRenderer(images=[sam_left]), 'right': StaticRenderer(images=[sam_right]), 'down': StaticRenderer(images=[sam_down]), 'up': StaticRenderer(images=[sam_up])}, path=path, start_frame=start_frame, stop_frame=stop_frame)

class Arrow(Sprite):
    """
    Sample arrow sprite - points where it is going.
    """

    def __init__(self, screen, path, colour=Screen.COLOUR_WHITE, start_frame=0, stop_frame=0):
        if False:
            while True:
                i = 10
        '\n        See :py:obj:`.Sprite` for details.\n        '
        super().__init__(screen, renderer_dict={'default': StaticRenderer(images=default_arrow, animation=_blink), 'left': StaticRenderer(images=[left_arrow]), 'right': StaticRenderer(images=[right_arrow]), 'down': StaticRenderer(images=[down_arrow]), 'up': StaticRenderer(images=[up_arrow])}, path=path, colour=colour, start_frame=start_frame, stop_frame=stop_frame)

class Plot(Sprite):
    """
    Sample Sprite that simply plots an "X" for each step in the path.  Useful
    for plotting a path to the screen.
    """

    def __init__(self, screen, path, colour=Screen.COLOUR_WHITE, start_frame=0, stop_frame=0):
        if False:
            print('Hello World!')
        '\n        See :py:obj:`.Sprite` for details.\n        '
        super().__init__(screen, renderer_dict={'default': StaticRenderer(images=['X'])}, path=path, colour=colour, clear=False, start_frame=start_frame, stop_frame=stop_frame)