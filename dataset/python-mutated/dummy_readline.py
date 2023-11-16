"""A dummy module with no effect for use on systems without readline."""
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional

def get_completer() -> Optional[Callable[[], str]]:
    if False:
        i = 10
        return i + 15
    'An empty implementation of readline.get_completer.'

def get_completer_delims() -> List[str]:
    if False:
        return 10
    'An empty implementation of readline.get_completer_delims.'
    return []

def parse_and_bind(unused_command: str) -> None:
    if False:
        i = 10
        return i + 15
    'An empty implementation of readline.parse_and_bind.'

def set_completer(unused_function: Optional[Callable[[], str]]=None) -> None:
    if False:
        return 10
    'An empty implementation of readline.set_completer.'

def set_completer_delims(unused_delims: Iterable[str]) -> None:
    if False:
        i = 10
        return i + 15
    'An empty implementation of readline.set_completer_delims.'