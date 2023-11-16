from typing import Optional, Set
from .base import ReplacementRule

class RuleValidator:

    def __init__(self, rule: ReplacementRule, *, char_domain: Optional[str]=None) -> None:
        if False:
            return 10
        self._rule = rule
        self._char_domain: Set[str] = set(char_domain) if char_domain else set('*/')

    def is_valid(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not self._is_all_stars()

    def _is_all_stars(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return set(self._rule) <= self._char_domain