"""Wrapper around argcomplete providing bug fixes and additional features."""
from __future__ import annotations
import argparse
import enum
import os
import typing as t

class Substitute:
    """Substitute for missing class which accepts all arguments."""

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        pass
try:
    import argcomplete
    try:
        from argcomplete.finders import CompletionFinder, default_validator
    except ImportError:
        from argcomplete import CompletionFinder, default_validator
    warn = argcomplete.warn
except ImportError:
    argcomplete = None
    CompletionFinder = Substitute
    default_validator = Substitute
    warn = Substitute

class CompType(enum.Enum):
    """
    Bash COMP_TYPE argument completion types.
    For documentation, see: https://www.gnu.org/software/bash/manual/html_node/Bash-Variables.html#index-COMP_005fTYPE
    """
    COMPLETION = '\t'
    '\n    Standard completion, typically triggered by a single tab.\n    '
    MENU_COMPLETION = '%'
    '\n    Menu completion, which cycles through each completion instead of showing a list.\n    For help using this feature, see: https://stackoverflow.com/questions/12044574/getting-complete-and-menu-complete-to-work-together\n    '
    LIST = '?'
    '\n    Standard list, typically triggered by a double tab.\n    '
    LIST_AMBIGUOUS = '!'
    '\n    Listing with `show-all-if-ambiguous` set.\n    For documentation, see https://www.gnu.org/software/bash/manual/html_node/Readline-Init-File-Syntax.html#index-show_002dall_002dif_002dambiguous\n    For additional details, see: https://unix.stackexchange.com/questions/614123/explanation-of-bash-completion-comp-type\n    '
    LIST_UNMODIFIED = '@'
    '\n    Listing with `show-all-if-unmodified` set.\n    For documentation, see https://www.gnu.org/software/bash/manual/html_node/Readline-Init-File-Syntax.html#index-show_002dall_002dif_002dunmodified\n    For additional details, see: : https://unix.stackexchange.com/questions/614123/explanation-of-bash-completion-comp-type\n    '

    @property
    def list_mode(self) -> bool:
        if False:
            while True:
                i = 10
        'True if completion is running in list mode, otherwise False.'
        return self in (CompType.LIST, CompType.LIST_AMBIGUOUS, CompType.LIST_UNMODIFIED)

def register_safe_action(action_type: t.Type[argparse.Action]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Register the given action as a safe action for argcomplete to use during completion if it is not already registered.'
    if argcomplete and action_type not in argcomplete.safe_actions:
        if isinstance(argcomplete.safe_actions, set):
            argcomplete.safe_actions.add(action_type)
        else:
            argcomplete.safe_actions += (action_type,)

def get_comp_type() -> t.Optional[CompType]:
    if False:
        i = 10
        return i + 15
    'Parse the COMP_TYPE environment variable (if present) and return the associated CompType enum value.'
    value = os.environ.get('COMP_TYPE')
    comp_type = CompType(chr(int(value))) if value else None
    return comp_type

class OptionCompletionFinder(CompletionFinder):
    """
    Custom completion finder for argcomplete.
    It provides support for running completion in list mode, which argcomplete natively handles the same as standard completion.
    """
    enabled = bool(argcomplete)

    def __init__(self, *args, validator=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        if validator:
            raise ValueError()
        self.comp_type = get_comp_type()
        self.list_mode = self.comp_type.list_mode if self.comp_type else False
        self.disable_completion_mangling = False
        finder = self

        def custom_validator(completion, prefix):
            if False:
                print('Hello World!')
            'Completion validator used to optionally bypass validation.'
            if finder.disable_completion_mangling:
                return True
            return default_validator(completion, prefix)
        super().__init__(*args, validator=custom_validator, **kwargs)

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        if self.enabled:
            super().__call__(*args, **kwargs)

    def quote_completions(self, completions, cword_prequote, last_wordbreak_pos):
        if False:
            return 10
        'Intercept default quoting behavior to optionally block mangling of completion entries.'
        if self.disable_completion_mangling:
            last_wordbreak_pos = None
        return super().quote_completions(completions, cword_prequote, last_wordbreak_pos)