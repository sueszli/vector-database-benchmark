from itertools import chain
import re
from robot.utils import NormalizedDict
from .stats import CombinedTagStat, TagStat
from .tags import TagPatterns

class TagStatistics:
    """Container for tag statistics."""

    def __init__(self, combined_stats):
        if False:
            for i in range(10):
                print('nop')
        self.tags = NormalizedDict(ignore='_')
        self.combined = combined_stats

    def visit(self, visitor):
        if False:
            print('Hello World!')
        visitor.visit_tag_statistics(self)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(sorted(chain(self.combined, self.tags.values())))

class TagStatisticsBuilder:

    def __init__(self, included=None, excluded=None, combined=None, docs=None, links=None):
        if False:
            while True:
                i = 10
        self._included = TagPatterns(included)
        self._excluded = TagPatterns(excluded)
        self._reserved = TagPatterns('robot:*')
        self._info = TagStatInfo(docs, links)
        self.stats = TagStatistics(self._info.get_combined_stats(combined))

    def add_test(self, test):
        if False:
            return 10
        self._add_tags_to_statistics(test)
        self._add_to_combined_statistics(test)

    def _add_tags_to_statistics(self, test):
        if False:
            for i in range(10):
                print('nop')
        for tag in test.tags:
            if self._is_included(tag) and (not self._suppress_reserved(tag)):
                if tag not in self.stats.tags:
                    self.stats.tags[tag] = self._info.get_stat(tag)
                self.stats.tags[tag].add_test(test)

    def _is_included(self, tag):
        if False:
            print('Hello World!')
        if self._included and tag not in self._included:
            return False
        return tag not in self._excluded

    def _suppress_reserved(self, tag):
        if False:
            print('Hello World!')
        return tag in self._reserved and tag not in self._included

    def _add_to_combined_statistics(self, test):
        if False:
            for i in range(10):
                print('nop')
        for stat in self.stats.combined:
            if stat.match(test.tags):
                stat.add_test(test)

class TagStatInfo:

    def __init__(self, docs=None, links=None):
        if False:
            i = 10
            return i + 15
        self._docs = [TagStatDoc(*doc) for doc in docs or []]
        self._links = [TagStatLink(*link) for link in links or []]

    def get_stat(self, tag):
        if False:
            return 10
        return TagStat(tag, self.get_doc(tag), self.get_links(tag))

    def get_combined_stats(self, combined=None):
        if False:
            while True:
                i = 10
        return [self._get_combined_stat(*comb) for comb in combined or []]

    def _get_combined_stat(self, pattern, name=None):
        if False:
            return 10
        name = name or pattern
        return CombinedTagStat(pattern, name, self.get_doc(name), self.get_links(name))

    def get_doc(self, tag):
        if False:
            for i in range(10):
                print('nop')
        return ' & '.join((doc.text for doc in self._docs if doc.match(tag)))

    def get_links(self, tag):
        if False:
            while True:
                i = 10
        return [link.get_link(tag) for link in self._links if link.match(tag)]

class TagStatDoc:

    def __init__(self, pattern, doc):
        if False:
            for i in range(10):
                print('nop')
        self._matcher = TagPatterns(pattern)
        self.text = doc

    def match(self, tag):
        if False:
            for i in range(10):
                print('nop')
        return self._matcher.match(tag)

class TagStatLink:
    _match_pattern_tokenizer = re.compile('(\\*|\\?+)')

    def __init__(self, pattern, link, title):
        if False:
            return 10
        self._regexp = self._get_match_regexp(pattern)
        self._link = link
        self._title = title.replace('_', ' ')

    def match(self, tag):
        if False:
            return 10
        return self._regexp.match(tag) is not None

    def get_link(self, tag):
        if False:
            for i in range(10):
                print('nop')
        match = self._regexp.match(tag)
        if not match:
            return None
        (link, title) = self._replace_groups(self._link, self._title, match)
        return (link, title)

    def _replace_groups(self, link, title, match):
        if False:
            while True:
                i = 10
        for (index, group) in enumerate(match.groups()):
            placefolder = '%%%d' % (index + 1)
            link = link.replace(placefolder, group)
            title = title.replace(placefolder, group)
        return (link, title)

    def _get_match_regexp(self, pattern):
        if False:
            return 10
        pattern = '^%s$' % ''.join(self._yield_match_pattern(pattern))
        return re.compile(pattern, re.IGNORECASE)

    def _yield_match_pattern(self, pattern):
        if False:
            return 10
        for token in self._match_pattern_tokenizer.split(pattern):
            if token.startswith('?'):
                yield ('(%s)' % ('.' * len(token)))
            elif token == '*':
                yield '(.*)'
            else:
                yield re.escape(token)