import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Union
from selenium.webdriver.firefox.webdriver import WebDriver
_ACCESSIBILITY_DIR = (Path(__file__).parent / 'accessibility-info').absolute()
_HTMLCS_RUNNER_CODE = "\nvar all_messages = [];\nconsole.log = msg => {\n    all_messages.push(msg);\n};\n\nHTMLCS_RUNNER.run('WCAG2AAA');\nreturn all_messages;\n"

class MessageType(Enum):
    ERROR = 1
    WARNING = 2
    NOTICE = 3

@dataclass
class Message:
    """Contains all of the information in a message emitted by HTML CodeSniffer."""
    principle_id: str
    message: str
    message_type: MessageType
    responsible_html: str
    selector: str
    element_type: str

    @staticmethod
    def from_output(output: str) -> 'Message':
        if False:
            while True:
                i = 10
        'Parses the output of htmlcs and returns an instance containing all data.\n\n        No processing is performed for flexibility.\n\n        Example message, post-split (note: contents of index 4 contains no newlines, but I had to\n        split it to keep the linter happy):\n\n        0: [HTMLCS] Error\n        1: WCAG2AAA.Principle1.Guideline1_3.1_3_1_AAA.G141\n        2: h2\n        3: #security-level-heading\n        4: The heading structure is not logically nested. This h2 element appears to be the\n           primary document heading, so should be an h1 element.\n        5: <h2 id="security-level-heading" hidden="">...</h2>\n        '
        fields = output.split('|')
        if 'Error' in fields[0]:
            message_type = MessageType.ERROR
        elif 'Warning' in fields[0]:
            message_type = MessageType.WARNING
        elif 'Notice' in fields[0]:
            message_type = MessageType.NOTICE
        return Message(message_type=message_type, principle_id=fields[1], element_type=fields[2], selector=fields[3], message=fields[4], responsible_html=fields[5])

    def __format__(self, _spec: str) -> str:
        if False:
            return 10
        newline = '\n'
        return f"\n{self.message_type}: {self.principle_id}\n    {self.message}\n\n    html:\n        {self.responsible_html.replace(newline, f'{newline}        ')}\n        "

def sniff_accessibility_issues(driver: WebDriver, locale: str, test_name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Runs accessibility sniffs on the driver's current page.\n\n    This function is responsible for injecting HTML CodeSniffer into the current page and writing\n    the results to a file. This way, test functions can focus on the setup required to navigate to\n    a particular URL (for example, logging in to get to the messages page).\n    "
    with open('/usr/local/lib/node_modules/html_codesniffer/build/HTMLCS.js') as htmlcs:
        html_codesniffer = htmlcs.read()
    errors_dir = _ACCESSIBILITY_DIR / locale / 'errors'
    errors_dir.mkdir(parents=True, exist_ok=True)
    reviews_dir = _ACCESSIBILITY_DIR / locale / 'reviews'
    reviews_dir.mkdir(parents=True, exist_ok=True)
    raw_messages = driver.execute_script(html_codesniffer + _HTMLCS_RUNNER_CODE)
    messages: Dict[str, List[Message]] = {'machine-verified': [], 'human-reviewed': []}
    for message in map(Message.from_output, raw_messages[:-1]):
        if message.message_type == MessageType.ERROR:
            messages['machine-verified'].append(message)
        else:
            messages['human-reviewed'].append(message)
    with open(errors_dir / f'{test_name}.txt', 'w') as error_file:
        for message in messages['machine-verified']:
            error_file.write(f'{message}')
    with open(reviews_dir / f'{test_name}.txt', 'w') as review_file:
        for message in messages['human-reviewed']:
            review_file.write(f'{message}')

def summarize_accessibility_results() -> None:
    if False:
        return 10
    'Creates a file containing summary information about the result of accessiblity sniffing\n\n    Note: This does not automatically run as part of the test suite, use\n          `make accessibility-summary` instead.\n    '
    try:
        summary: Dict[str, Dict[str, Dict[str, Union[int, bool]]]] = {}
        for out_filename in os.listdir(_ACCESSIBILITY_DIR / 'en_US' / 'reviews'):
            summary[out_filename] = {'reviews': {'count': 0, 'locale_differs': False}, 'errors': {'count': 0, 'locale_differs': False}}
            for message_type in ['reviews', 'errors']:
                outputs: Dict[str, Dict[str, List[str]]] = {}
                for locale in ['en_US', 'ar']:
                    outputs[locale] = {}
                    with open(_ACCESSIBILITY_DIR / locale / message_type / out_filename) as out_file:
                        outputs[locale][message_type] = [line for line in out_file.readlines() if 'MessageType.' in line]
                summary[out_filename][message_type]['count'] = len(outputs['en_US'][message_type])
                summary[out_filename][message_type]['locale_differs'] = outputs['en_US'][message_type] != outputs['ar'][message_type]
        with open(_ACCESSIBILITY_DIR / 'summary.txt', 'w') as summary_file:
            for name in sorted(summary.keys()):
                summary_file.write(name + ':\n')
                for message_type in ['errors', 'reviews']:
                    summary_file.write(f"\t{message_type}: {summary[name][message_type]['count']}\n")
                    if summary[name][message_type]['locale_differs']:
                        summary_file.write(f'\t        NOTE: {message_type} differ by locale\n')
                summary_file.write('\n')
    except FileNotFoundError:
        print(f'ERROR: Run `make test TESTFILES={os.path.dirname(_ACCESSIBILITY_DIR)}` before running `make accessibility-summary`')