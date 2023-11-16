import re
import sqlparse
from sqlparse.tokens import Name
from collections import defaultdict
from .pgliterals.main import get_literals
white_space_regex = re.compile('\\s+', re.MULTILINE)

def _compile_regex(keyword):
    if False:
        for i in range(10):
            print('nop')
    pattern = '\\b' + white_space_regex.sub('\\\\s+', keyword) + '\\b'
    return re.compile(pattern, re.MULTILINE | re.IGNORECASE)
keywords = get_literals('keywords')
keyword_regexs = {kw: _compile_regex(kw) for kw in keywords}

class PrevalenceCounter:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.keyword_counts = defaultdict(int)
        self.name_counts = defaultdict(int)

    def update(self, text):
        if False:
            print('Hello World!')
        self.update_keywords(text)
        self.update_names(text)

    def update_names(self, text):
        if False:
            for i in range(10):
                print('nop')
        for parsed in sqlparse.parse(text):
            for token in parsed.flatten():
                if token.ttype in Name:
                    self.name_counts[token.value] += 1

    def clear_names(self):
        if False:
            while True:
                i = 10
        self.name_counts = defaultdict(int)

    def update_keywords(self, text):
        if False:
            return 10
        for (keyword, regex) in keyword_regexs.items():
            for _ in regex.finditer(text):
                self.keyword_counts[keyword] += 1

    def keyword_count(self, keyword):
        if False:
            i = 10
            return i + 15
        return self.keyword_counts[keyword]

    def name_count(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self.name_counts[name]