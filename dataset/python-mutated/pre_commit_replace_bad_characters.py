from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import NamedTuple
from rich.console import Console
if __name__ != '__main__':
    raise Exception(f'This file is intended to be executed as an executable program. You cannot use it as a module.To run this script, run the {__file__} command')
console = Console(width=400, color_system='standard')

class RegexpSpec(NamedTuple):
    regexp: str
    replacement: str
    description: str
REPLACEMENTS: list[RegexpSpec] = [RegexpSpec(regexp='\\t', replacement='    ', description='<TAB> with 4 spaces'), RegexpSpec(regexp='\\u00A0', replacement=' ', description='&nbsp with space'), RegexpSpec(regexp='\\u2018', replacement="'", description='left single quotation with straight one'), RegexpSpec(regexp='\\u2019', replacement="'", description='right single quotation with straight one'), RegexpSpec(regexp='\\u201C', replacement='"', description='left double quotation with straight one'), RegexpSpec(regexp='\\u201D', replacement='"', description='right double quotation with straight one')]

def main() -> int:
    if False:
        return 10
    total_count_changes = 0
    matches = [re.compile(spec.regexp) for spec in REPLACEMENTS]
    for file_string in sys.argv:
        count_changes = 0
        path = Path(file_string)
        text = path.read_text()
        for (match, spec) in zip(matches, REPLACEMENTS):
            (text, new_count_changes) = match.subn(spec.replacement, text)
            if new_count_changes:
                console.print(f'[yellow] Performed {new_count_changes} replacements of {spec.description}[/]: {path}')
            count_changes += new_count_changes
        if count_changes:
            path.write_text(text)
        total_count_changes += count_changes
    return 1 if total_count_changes else 0
sys.exit(main())