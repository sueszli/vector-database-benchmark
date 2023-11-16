import logging
from collections import OrderedDict
from pprint import pformat
from typing import Dict, Union
from ludwig.api_annotations import DeveloperAPI
logger = logging.getLogger(__name__)

@DeveloperAPI
def get_logging_level_registry() -> Dict[str, int]:
    if False:
        while True:
            i = 10
    return {'critical': logging.CRITICAL, 'error': logging.ERROR, 'warning': logging.WARNING, 'info': logging.INFO, 'debug': logging.DEBUG, 'notset': logging.NOTSET}

@DeveloperAPI
def get_logo(message, ludwig_version):
    if False:
        for i in range(10):
            print('nop')
    return '\n'.join(['███████████████████████', '█ █ █ █  ▜█ █ █ █ █   █', '█ █ █ █ █ █ █ █ █ █ ███', '█ █   █ █ █ █ █ █ █ ▌ █', '█ █████ █ █ █ █ █ █ █ █', '█     █  ▟█     █ █   █', '███████████████████████', f'ludwig v{ludwig_version} - {message}', ''])

@DeveloperAPI
def print_ludwig(message, ludwig_version):
    if False:
        for i in range(10):
            print('nop')
    logger.info(get_logo(message, ludwig_version))

@DeveloperAPI
def print_boxed(text, print_fun=logger.info):
    if False:
        return 10
    box_width = len(text) + 2
    print_fun('')
    print_fun('╒{}╕'.format('═' * box_width))
    print_fun(f'│ {text.upper()} │')
    print_fun('╘{}╛'.format('═' * box_width))
    print_fun('')

@DeveloperAPI
def repr_ordered_dict(d: OrderedDict):
    if False:
        for i in range(10):
            print('nop')
    return '{' + ',\n  '.join((f'{x}: {pformat(y, indent=4)}' for (x, y) in d.items())) + '}'

@DeveloperAPI
def query_yes_no(question: str, default: Union[str, None]='yes'):
    if False:
        i = 10
        return i + 15
    'Ask a yes/no question via raw_input() and return their answer.\n\n    Args:\n        question: String presented to the user\n        default: The presumed answer from the user. Must be "yes", "no", or None (Answer is required)\n\n    Returns: Boolean based on prompt response\n    '
    valid = {'yes': True, 'y': True, 'ye': True, 'no': False, 'n': False}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    while True:
        logger.info(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            logger.info("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")