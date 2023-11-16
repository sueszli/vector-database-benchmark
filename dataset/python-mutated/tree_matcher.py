"""Tree matcher based on Lark grammar"""
import re
from collections import defaultdict
from . import Tree, Token
from .common import ParserConf
from .parsers import earley
from .grammar import Rule, Terminal, NonTerminal

def is_discarded_terminal(t):
    if False:
        print('Hello World!')
    return t.is_term and t.filter_out

class _MakeTreeMatch:

    def __init__(self, name, expansion):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.expansion = expansion

    def __call__(self, args):
        if False:
            return 10
        t = Tree(self.name, args)
        t.meta.match_tree = True
        t.meta.orig_expansion = self.expansion
        return t

def _best_from_group(seq, group_key, cmp_key):
    if False:
        return 10
    d = {}
    for item in seq:
        key = group_key(item)
        if key in d:
            v1 = cmp_key(item)
            v2 = cmp_key(d[key])
            if v2 > v1:
                d[key] = item
        else:
            d[key] = item
    return list(d.values())

def _best_rules_from_group(rules):
    if False:
        i = 10
        return i + 15
    rules = _best_from_group(rules, lambda r: r, lambda r: -len(r.expansion))
    rules.sort(key=lambda r: len(r.expansion))
    return rules

def _match(term, token):
    if False:
        while True:
            i = 10
    if isinstance(token, Tree):
        (name, _args) = parse_rulename(term.name)
        return token.data == name
    elif isinstance(token, Token):
        return term == Terminal(token.type)
    assert False, (term, token)

def make_recons_rule(origin, expansion, old_expansion):
    if False:
        return 10
    return Rule(origin, expansion, alias=_MakeTreeMatch(origin.name, old_expansion))

def make_recons_rule_to_term(origin, term):
    if False:
        print('Hello World!')
    return make_recons_rule(origin, [Terminal(term.name)], [term])

def parse_rulename(s):
    if False:
        while True:
            i = 10
    'Parse rule names that may contain a template syntax (like rule{a, b, ...})'
    (name, args_str) = re.match('(\\w+)(?:{(.+)})?', s).groups()
    args = args_str and [a.strip() for a in args_str.split(',')]
    return (name, args)

class ChildrenLexer:

    def __init__(self, children):
        if False:
            for i in range(10):
                print('nop')
        self.children = children

    def lex(self, parser_state):
        if False:
            i = 10
            return i + 15
        return self.children

class TreeMatcher:
    """Match the elements of a tree node, based on an ontology
    provided by a Lark grammar.

    Supports templates and inlined rules (`rule{a, b,..}` and `_rule`)

    Initialize with an instance of Lark.
    """

    def __init__(self, parser):
        if False:
            return 10
        assert not parser.options.maybe_placeholders
        (self.tokens, rules, _extra) = parser.grammar.compile(parser.options.start, set())
        self.rules_for_root = defaultdict(list)
        self.rules = list(self._build_recons_rules(rules))
        self.rules.reverse()
        self.rules = _best_rules_from_group(self.rules)
        self.parser = parser
        self._parser_cache = {}

    def _build_recons_rules(self, rules):
        if False:
            while True:
                i = 10
        'Convert tree-parsing/construction rules to tree-matching rules'
        expand1s = {r.origin for r in rules if r.options.expand1}
        aliases = defaultdict(list)
        for r in rules:
            if r.alias:
                aliases[r.origin].append(r.alias)
        rule_names = {r.origin for r in rules}
        nonterminals = {sym for sym in rule_names if sym.name.startswith('_') or sym in expand1s or sym in aliases}
        seen = set()
        for r in rules:
            recons_exp = [sym if sym in nonterminals else Terminal(sym.name) for sym in r.expansion if not is_discarded_terminal(sym)]
            if recons_exp == [r.origin] and r.alias is None:
                continue
            sym = NonTerminal(r.alias) if r.alias else r.origin
            rule = make_recons_rule(sym, recons_exp, r.expansion)
            if sym in expand1s and len(recons_exp) != 1:
                self.rules_for_root[sym.name].append(rule)
                if sym.name not in seen:
                    yield make_recons_rule_to_term(sym, sym)
                    seen.add(sym.name)
            elif sym.name.startswith('_') or sym in expand1s:
                yield rule
            else:
                self.rules_for_root[sym.name].append(rule)
        for (origin, rule_aliases) in aliases.items():
            for alias in rule_aliases:
                yield make_recons_rule_to_term(origin, NonTerminal(alias))
            yield make_recons_rule_to_term(origin, origin)

    def match_tree(self, tree, rulename):
        if False:
            print('Hello World!')
        "Match the elements of `tree` to the symbols of rule `rulename`.\n\n        Parameters:\n            tree (Tree): the tree node to match\n            rulename (str): The expected full rule name (including template args)\n\n        Returns:\n            Tree: an unreduced tree that matches `rulename`\n\n        Raises:\n            UnexpectedToken: If no match was found.\n\n        Note:\n            It's the callers' responsibility match the tree recursively.\n        "
        if rulename:
            (name, _args) = parse_rulename(rulename)
            assert tree.data == name
        else:
            rulename = tree.data
        try:
            parser = self._parser_cache[rulename]
        except KeyError:
            rules = self.rules + _best_rules_from_group(self.rules_for_root[rulename])
            callbacks = {rule: rule.alias for rule in rules}
            conf = ParserConf(rules, callbacks, [rulename])
            parser = earley.Parser(self.parser.lexer_conf, conf, _match, resolve_ambiguity=True)
            self._parser_cache[rulename] = parser
        unreduced_tree = parser.parse(ChildrenLexer(tree.children), rulename)
        assert unreduced_tree.data == rulename
        return unreduced_tree