from typing import Dict, Optional
from returns.maybe import Nothing, Some, maybe

@maybe
def _function(hashmap: Dict[str, str], key: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    return hashmap.get(key, None)

def test_maybe_some():
    if False:
        while True:
            i = 10
    'Ensures that maybe decorator works correctly for some case.'
    assert _function({'a': 'b'}, 'a') == Some('b')

def test_maybe_nothing():
    if False:
        print('Hello World!')
    'Ensures that maybe decorator works correctly for nothing case.'
    assert _function({'a': 'b'}, 'c') == Nothing