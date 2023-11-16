from asciimatics.effects import Cycle, Snow, Print
from asciimatics.renderers import FigletText, StaticRenderer
from asciimatics.scene import Scene
from asciimatics.screen import Screen
from asciimatics.exceptions import ResizeScreenError
import sys
tree = ('\n       ${3,1}*\n      / \\\n     /${1}o${2}  \\\n    /_   _\\\n     /   \\${4}b\n    /     \\\n   /   ${1}o${2}   \\\n  /__     __\\\n  ${1}d${2} / ${4}o${2}   \\\n   /       \\\n  / ${4}o     ${1}o${2}.\\\n /___________\\\n      ${3}|||\n      ${3}|||\n', '\n       ${3}*\n      / \\\n     /${1}o${2}  \\\n    /_   _\\\n     /   \\${4}b\n    /     \\\n   /   ${1}o${2}   \\\n  /__     __\\\n  ${1}d${2} / ${4}o${2}   \\\n   /       \\\n  / ${4}o     ${1}o${2} \\\n /___________\\\n      ${3}|||\n      ${3}|||\n')

def demo(screen):
    if False:
        i = 10
        return i + 15
    effects = [Print(screen, StaticRenderer(images=tree), x=screen.width - 15, y=screen.height - 15, colour=Screen.COLOUR_GREEN), Snow(screen), Cycle(screen, FigletText('HAPPY'), screen.height // 2 - 6, start_frame=300), Cycle(screen, FigletText('XMAS!'), screen.height // 2 + 1, start_frame=300)]
    screen.play([Scene(effects, -1)], stop_on_resize=True)
while True:
    try:
        Screen.wrapper(demo)
        sys.exit(0)
    except ResizeScreenError:
        pass