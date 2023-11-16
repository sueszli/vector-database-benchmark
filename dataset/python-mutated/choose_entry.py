import re
from typing import Any, Generator, List, Type
from .cli_stub import HintsCLIOptions
from .typing import MarkType

def mark(text: str, args: HintsCLIOptions, Mark: Type[MarkType], extra_cli_args: List[str], *a: Any) -> Generator[MarkType, None, None]:
    if False:
        for i in range(10):
            print('nop')
    idx = 0
    found_start_line = False
    for m in re.finditer('(?m)^.+$', text):
        (start, end) = m.span()
        line = text[start:end].replace('\x00', '').replace('\n', '')
        if line == ' ':
            found_start_line = True
            continue
        if line.startswith(': '):
            yield Mark(idx, start, end, line, {'index': idx})
            idx += 1
        elif found_start_line:
            idx += 1