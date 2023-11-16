from collections.abc import Sequence
from robot.result import Keyword
from robot.utils import MultiMatcher

class ExpandKeywordMatcher:

    def __init__(self, expand_keywords: 'str|Sequence[str]'):
        if False:
            return 10
        self.matched_ids: 'list[str]' = []
        if not expand_keywords:
            expand_keywords = []
        elif isinstance(expand_keywords, str):
            expand_keywords = [expand_keywords]
        names = [n[5:] for n in expand_keywords if n[:5].lower() == 'name:']
        tags = [p[4:] for p in expand_keywords if p[:4].lower() == 'tag:']
        self._match_name = MultiMatcher(names).match
        self._match_tags = MultiMatcher(tags).match_any

    def match(self, kw: Keyword):
        if False:
            for i in range(10):
                print('nop')
        if (self._match_name(kw.full_name or '') or self._match_tags(kw.tags)) and (not kw.not_run):
            self.matched_ids.append(kw.id)