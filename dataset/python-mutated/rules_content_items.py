import re
from re import Pattern
from tribler.core.components.knowledge.rules.tag_rules_base import Rule, RulesList
space = '[-\\._\\s]'
two_digit_version = '(\\d{1,2}(?:\\.\\d{1,2})?)'

def pattern(linux_distribution: str) -> Pattern:
    if False:
        return 10
    return re.compile(f'{linux_distribution}{space}*{two_digit_version}', flags=re.IGNORECASE)
content_items_rules: RulesList = [Rule(patterns=[pattern('ubuntu')], actions=[lambda s: f'Ubuntu {s}']), Rule(patterns=[pattern('debian')], actions=[lambda s: f'Debian {s}']), Rule(patterns=[re.compile(f'linux{space}*mint{space}*{two_digit_version}', flags=re.IGNORECASE)], actions=[lambda s: f'Linux Mint {s}'])]