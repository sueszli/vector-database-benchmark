import sys
from .conf import settings
from .types import Rule
from .system import Path
from . import logs

def get_loaded_rules(rules_paths):
    if False:
        for i in range(10):
            print('nop')
    'Yields all available rules.\n\n    :type rules_paths: [Path]\n    :rtype: Iterable[Rule]\n\n    '
    for path in rules_paths:
        if path.name != '__init__.py':
            rule = Rule.from_path(path)
            if rule and rule.is_enabled:
                yield rule

def get_rules_import_paths():
    if False:
        print('Hello World!')
    'Yields all rules import paths.\n\n    :rtype: Iterable[Path]\n\n    '
    yield Path(__file__).parent.joinpath('rules')
    yield settings.user_dir.joinpath('rules')
    for path in sys.path:
        for contrib_module in Path(path).glob('thefuck_contrib_*'):
            contrib_rules = contrib_module.joinpath('rules')
            if contrib_rules.is_dir():
                yield contrib_rules

def get_rules():
    if False:
        print('Hello World!')
    'Returns all enabled rules.\n\n    :rtype: [Rule]\n\n    '
    paths = [rule_path for path in get_rules_import_paths() for rule_path in sorted(path.glob('*.py'))]
    return sorted(get_loaded_rules(paths), key=lambda rule: rule.priority)

def organize_commands(corrected_commands):
    if False:
        while True:
            i = 10
    'Yields sorted commands without duplicates.\n\n    :type corrected_commands: Iterable[thefuck.types.CorrectedCommand]\n    :rtype: Iterable[thefuck.types.CorrectedCommand]\n\n    '
    try:
        first_command = next(corrected_commands)
        yield first_command
    except StopIteration:
        return
    without_duplicates = {command for command in sorted(corrected_commands, key=lambda command: command.priority) if command != first_command}
    sorted_commands = sorted(without_duplicates, key=lambda corrected_command: corrected_command.priority)
    logs.debug(u'Corrected commands: {}'.format(', '.join((u'{}'.format(cmd) for cmd in [first_command] + sorted_commands))))
    for command in sorted_commands:
        yield command

def get_corrected_commands(command):
    if False:
        while True:
            i = 10
    'Returns generator with sorted and unique corrected commands.\n\n    :type command: thefuck.types.Command\n    :rtype: Iterable[thefuck.types.CorrectedCommand]\n\n    '
    corrected_commands = (corrected for rule in get_rules() if rule.is_match(command) for corrected in rule.get_corrected_commands(command))
    return organize_commands(corrected_commands)