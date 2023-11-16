import argparse
from copy import copy
from typing import Dict
try:

    def convert_to_list(tuple):
        if False:
            print('Hello World!')
        return [item for item in tuple]
    from shtab import add_argument_to, FILE, DIR
    defaults = convert_to_list(add_argument_to.__defaults__)
    defaults[-1] = {'zsh': '\n# https://github.com/zsh-users/zsh/blob/19390a1ba8dc983b0a1379058e90cd51ce156815/Completion/Unix/Command/_rm#L72-L74\n_trash_files() {\n  (( CURRENT > 0 )) && line[CURRENT]=()\n  line=( ${line//(#m)[\\[\\]()\\\\*?#<>~\\^\\|]/\\\\$MATCH} )\n  _files -F line\n}\n'}
    add_argument_to.__defaults__ = tuple(defaults)
    TRASH_FILES = copy(FILE)
    TRASH_DIRS = copy(DIR)

    def complete_with(completion, action):
        if False:
            print('Hello World!')
        action.complete = completion
except ImportError:
    from argparse import Action
    TRASH_FILES = TRASH_DIRS = {}

    class PrintCompletionAction(Action):

        def __call__(self, parser, namespace, values, option_string=None):
            if False:
                return 10
            print('Please install shtab firstly!')
            parser.exit(0)

    def add_argument_to(parser, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        Action.complete = None
        parser.add_argument('--print-completion', choices=['bash', 'zsh', 'tcsh'], action=PrintCompletionAction, help='print shell completion script')
        return parser

    def complete_with(completion, action):
        if False:
            return 10
        pass
TRASH_FILES.update({'zsh': '_trash_files'})
TRASH_DIRS.update({'zsh': '(${$(trash-list --trash-dirs)#parent_*:})'})