import re
from collections.abc import Mapping
ELEMENT_SELECTOR = re.compile('^([\\w-]+)$')
ELEMENT_WITH_ATTR_SELECTOR = re.compile('^([\\w-]+)\\[([\\w-]+)\\]$')
ELEMENT_WITH_ATTR_EXACT_SINGLE_QUOTE_SELECTOR = re.compile("^([\\w-]+)\\[([\\w-]+)='(.*)'\\]$")
ELEMENT_WITH_ATTR_EXACT_DOUBLE_QUOTE_SELECTOR = re.compile('^([\\w-]+)\\[([\\w-]+)="(.*)"\\]$')
ELEMENT_WITH_ATTR_EXACT_UNQUOTED_SELECTOR = re.compile('^([\\w-]+)\\[([\\w-]+)=([\\w-]+)\\]$')

class HTMLRuleset:
    """
    Maintains a set of rules for matching HTML elements.
    Each rule defines a mapping from a CSS-like selector to an arbitrary result object.

    The following forms of rule are currently supported:
    'a' = matches any <a> element
    'a[href]' = matches any <a> element with an 'href' attribute
    'a[linktype="page"]' = matches any <a> element with a 'linktype' attribute equal to 'page'
    """

    def __init__(self, rules=None):
        if False:
            for i in range(10):
                print('nop')
        self.element_rules = {}
        if rules:
            self.add_rules(rules)

    def add_rules(self, rules):
        if False:
            while True:
                i = 10
        if isinstance(rules, Mapping):
            rules = rules.items()
        for (selector, result) in rules:
            self.add_rule(selector, result)

    def _add_element_rule(self, name, result):
        if False:
            i = 10
            return i + 15
        rules = self.element_rules.setdefault(name, [])
        rules.append((2, lambda attrs: True, result))
        rules.sort(key=lambda t: t[0])

    def _add_element_with_attr_rule(self, name, attr, result):
        if False:
            i = 10
            return i + 15
        rules = self.element_rules.setdefault(name, [])
        rules.append((1, lambda attrs: attr in attrs, result))
        rules.sort(key=lambda t: t[0])

    def _add_element_with_attr_exact_rule(self, name, attr, value, result):
        if False:
            i = 10
            return i + 15
        rules = self.element_rules.setdefault(name, [])
        rules.append((1, lambda attrs: attr in attrs and attrs[attr] == value, result))
        rules.sort(key=lambda t: t[0])

    def add_rule(self, selector, result):
        if False:
            for i in range(10):
                print('nop')
        match = ELEMENT_SELECTOR.match(selector)
        if match:
            name = match.group(1)
            self._add_element_rule(name, result)
            return
        match = ELEMENT_WITH_ATTR_SELECTOR.match(selector)
        if match:
            (name, attr) = match.groups()
            self._add_element_with_attr_rule(name, attr, result)
            return
        for regex in (ELEMENT_WITH_ATTR_EXACT_SINGLE_QUOTE_SELECTOR, ELEMENT_WITH_ATTR_EXACT_DOUBLE_QUOTE_SELECTOR, ELEMENT_WITH_ATTR_EXACT_UNQUOTED_SELECTOR):
            match = regex.match(selector)
            if match:
                (name, attr, value) = match.groups()
                self._add_element_with_attr_exact_rule(name, attr, value, result)
                return

    def match(self, name, attrs):
        if False:
            return 10
        '\n        Look for a rule matching an HTML element with the given name and attribute dict,\n        and return the corresponding result object. If no rule matches, return None.\n        If multiple rules match, the one chosen is undetermined.\n        '
        try:
            rules_to_test = self.element_rules[name]
        except KeyError:
            return None
        for (precedence, attr_check, result) in rules_to_test:
            if attr_check(attrs):
                return result