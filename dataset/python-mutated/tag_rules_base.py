from dataclasses import dataclass, field
from typing import AnyStr, Callable, Iterable, Optional, Pattern, Sequence
from tribler.core.components.knowledge.community.knowledge_validator import is_valid_resource

@dataclass
class Rule:
    patterns: Sequence[Pattern[AnyStr]] = field(default_factory=lambda : [])
    actions: Sequence[Callable[[str], str]] = field(default_factory=lambda : [])
RulesList = Sequence[Rule]

def extract_tags(text: str, rules: Optional[RulesList]=None) -> Iterable[str]:
    if False:
        while True:
            i = 10
    ' Extract tags by using the given rules.\n\n    Rules are represented by an array of an array of regexes.\n    Each rule contains one or more regex expressions.\n\n    During the `text` processing, each rule will be applied to the `text` value.\n    All extracted tags will be returned.\n\n    During application of the particular rule, `text` will be split into\n    tokens by application of the first regex expression. Then, second regex\n    expression will be applied to each tokens that were extracted on the\n    previous step.\n    This process will be repeated until regex expression ends.\n\n    For the each string result the action will be applied.\n    '
    rules = rules or []
    for rule in rules:
        text_set = {text}
        for regex in rule.patterns:
            next_text_set = set()
            for token in text_set:
                for match in regex.finditer(token):
                    next_text_set |= set(match.groups())
            text_set = next_text_set
        for action in rule.actions:
            text_set = map(action, text_set)
        yield from text_set

def extract_only_valid_tags(text: str, rules: Optional[RulesList]=None) -> Iterable[str]:
    if False:
        print('Hello World!')
    for tag in extract_tags(text, rules):
        tag = tag.lower()
        if is_valid_resource(tag):
            yield tag