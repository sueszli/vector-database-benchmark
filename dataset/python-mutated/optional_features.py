from typing import Set
_optional_features: Set[str] = set()

def register(feature: str) -> None:
    if False:
        while True:
            i = 10
    'Register an optional feature.'
    _optional_features.add(feature)

def has(feature: str) -> bool:
    if False:
        print('Hello World!')
    'Check if an optional feature is registered.'
    return feature in _optional_features