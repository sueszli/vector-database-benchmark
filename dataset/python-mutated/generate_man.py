import os.path
import sys
import textwrap
from build_manpages.manpage import Manpage
from pipx.main import get_command_parser

def main():
    if False:
        while True:
            i = 10
    sys.argv[0] = 'pipx'
    parser = get_command_parser()
    parser.man_short_description = parser.description.splitlines()[1]
    manpage = Manpage(parser)
    body = str(manpage)
    body = body.replace(os.path.expanduser('~').replace('-', '\\-'), '~')
    body += textwrap.dedent('\n        .SH AUTHORS\n        .IR pipx (1)\n        was written by Chad Smith and contributors.\n        The project can be found online at\n        .UR https://pypa.github.io/pipx/\n        .UE\n        .SH SEE ALSO\n        .IR pip (1),\n        .IR virtualenv (1)\n        ')
    with open('pipx.1', 'w') as f:
        f.write(body)
if __name__ == '__main__':
    main()