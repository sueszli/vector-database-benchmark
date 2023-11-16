import argparse
import re
import sys
from typing import Dict, List
import requests
BASE_URL = 'https://api.github.com/repos/zauberzeug/nicegui'
parser = argparse.ArgumentParser(description='Fetch the content of a milestone from a GitHub repo.')
parser.add_argument('milestone_title', help='Title of the milestone to fetch.')
args = parser.parse_args()
milestone_title: str = args.milestone_title
milestones = requests.get(f'{BASE_URL}/milestones', timeout=5).json()
matching_milestones = [milestone for milestone in milestones if milestone['title'] == milestone_title]
if not matching_milestones:
    print(f'Milestone "{milestone_title}" not found!')
    sys.exit(1)
milestone_number = matching_milestones[0]['number']

def link(number: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    escape_mask = '\x1b]8;{};{}\x1b\\{}\x1b]8;;\x1b\\'
    return escape_mask.format('', f'https://github.com/zauberzeug/nicegui/issues/{number}', f'#{number}')
issues = requests.get(f'{BASE_URL}/issues?milestone={milestone_number}&state=all', timeout=5).json()
notes: Dict[str, List[str]] = {'New features and enhancements': [], 'Bugfixes': [], 'Documentation': [], 'Others': []}
for issue in issues:
    title: str = issue['title']
    user: str = issue['user']['login']
    body: str = issue['body']
    labels: list[str] = [label['name'] for label in issue['labels']]
    number_patterns = ['#(\\d+)', 'https://github.com/zauberzeug/nicegui/(?:issues|discussions|pulls)/(\\d+)']
    numbers = [issue['number']] + [int(match) for pattern in number_patterns for match in re.findall(pattern, body)]
    numbers_str = ', '.join((link(number) for number in sorted(numbers)))
    note = f'{title.strip()} ({numbers_str} by @{user})'
    if 'bug' in labels:
        notes['Bugfixes'].append(note)
    elif 'enhancement' in labels:
        notes['New features and enhancements'].append(note)
    elif 'documentation' in labels:
        notes['Documentation'].append(note)
    else:
        notes['Others'].append(note)
for (title, lines) in notes.items():
    if not lines:
        continue
    print(f'### {title}')
    print()
    for line in lines:
        print(f'- {line}')
    print()