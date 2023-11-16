"""This module implements an Earley parser.

The core Earley algorithm used here is based on Elizabeth Scott's implementation, here:
    https://www.sciencedirect.com/science/article/pii/S1571066108001497

That is probably the best reference for understanding the algorithm here.

The Earley parser outputs an SPPF-tree as per that document. The SPPF tree format
is explained here: https://lark-parser.readthedocs.io/en/latest/_static/sppf/sppf.html
"""
from typing import TYPE_CHECKING, Callable, Optional, List, Any
from collections import deque
from ..lexer import Token
from ..tree import Tree
from ..exceptions import UnexpectedEOF, UnexpectedToken
from ..utils import logger, OrderedSet
from .grammar_analysis import GrammarAnalyzer
from ..grammar import NonTerminal
from .earley_common import Item
from .earley_forest import ForestSumVisitor, SymbolNode, StableSymbolNode, TokenNode, ForestToParseTree
if TYPE_CHECKING:
    from ..common import LexerConf, ParserConf

class Parser:
    lexer_conf: 'LexerConf'
    parser_conf: 'ParserConf'
    debug: bool

    def __init__(self, lexer_conf: 'LexerConf', parser_conf: 'ParserConf', term_matcher: Callable, resolve_ambiguity: bool=True, debug: bool=False, tree_class: Optional[Callable[[str, List], Any]]=Tree, ordered_sets: bool=True):
        if False:
            return 10
        analysis = GrammarAnalyzer(parser_conf)
        self.lexer_conf = lexer_conf
        self.parser_conf = parser_conf
        self.resolve_ambiguity = resolve_ambiguity
        self.debug = debug
        self.Tree = tree_class
        self.Set = OrderedSet if ordered_sets else set
        self.SymbolNode = StableSymbolNode if ordered_sets else SymbolNode
        self.FIRST = analysis.FIRST
        self.NULLABLE = analysis.NULLABLE
        self.callbacks = parser_conf.callbacks
        self.predictions = {}
        self.TERMINALS = {sym for r in parser_conf.rules for sym in r.expansion if sym.is_term}
        self.NON_TERMINALS = {sym for r in parser_conf.rules for sym in r.expansion if not sym.is_term}
        self.forest_sum_visitor = None
        for rule in parser_conf.rules:
            if rule.origin not in self.predictions:
                self.predictions[rule.origin] = [x.rule for x in analysis.expand_rule(rule.origin)]
            if self.forest_sum_visitor is None and rule.options.priority is not None:
                self.forest_sum_visitor = ForestSumVisitor
        if self.lexer_conf.lexer_type != 'basic' and self.forest_sum_visitor is None:
            for term in self.lexer_conf.terminals:
                if term.priority:
                    self.forest_sum_visitor = ForestSumVisitor
                    break
        self.term_matcher = term_matcher

    def predict_and_complete(self, i, to_scan, columns, transitives):
        if False:
            i = 10
            return i + 15
        'The core Earley Predictor and Completer.\n\n        At each stage of the input, we handling any completed items (things\n        that matched on the last cycle) and use those to predict what should\n        come next in the input stream. The completions and any predicted\n        non-terminals are recursively processed until we reach a set of,\n        which can be added to the scan list for the next scanner cycle.'
        node_cache = {}
        held_completions = {}
        column = columns[i]
        items = deque(column)
        while items:
            item = items.pop()
            if item.is_complete:
                if item.node is None:
                    label = (item.s, item.start, i)
                    item.node = node_cache[label] if label in node_cache else node_cache.setdefault(label, self.SymbolNode(*label))
                    item.node.add_family(item.s, item.rule, item.start, None, None)
                if item.rule.origin in transitives[item.start]:
                    transitive = transitives[item.start][item.s]
                    if transitive.previous in transitives[transitive.column]:
                        root_transitive = transitives[transitive.column][transitive.previous]
                    else:
                        root_transitive = transitive
                    new_item = Item(transitive.rule, transitive.ptr, transitive.start)
                    label = (root_transitive.s, root_transitive.start, i)
                    new_item.node = node_cache[label] if label in node_cache else node_cache.setdefault(label, self.SymbolNode(*label))
                    new_item.node.add_path(root_transitive, item.node)
                    if new_item.expect in self.TERMINALS:
                        to_scan.add(new_item)
                    elif new_item not in column:
                        column.add(new_item)
                        items.append(new_item)
                else:
                    is_empty_item = item.start == i
                    if is_empty_item:
                        held_completions[item.rule.origin] = item.node
                    originators = [originator for originator in columns[item.start] if originator.expect is not None and originator.expect == item.s]
                    for originator in originators:
                        new_item = originator.advance()
                        label = (new_item.s, originator.start, i)
                        new_item.node = node_cache[label] if label in node_cache else node_cache.setdefault(label, self.SymbolNode(*label))
                        new_item.node.add_family(new_item.s, new_item.rule, i, originator.node, item.node)
                        if new_item.expect in self.TERMINALS:
                            to_scan.add(new_item)
                        elif new_item not in column:
                            column.add(new_item)
                            items.append(new_item)
            elif item.expect in self.NON_TERMINALS:
                new_items = []
                for rule in self.predictions[item.expect]:
                    new_item = Item(rule, 0, i)
                    new_items.append(new_item)
                if item.expect in held_completions:
                    new_item = item.advance()
                    label = (new_item.s, item.start, i)
                    new_item.node = node_cache[label] if label in node_cache else node_cache.setdefault(label, self.SymbolNode(*label))
                    new_item.node.add_family(new_item.s, new_item.rule, new_item.start, item.node, held_completions[item.expect])
                    new_items.append(new_item)
                for new_item in new_items:
                    if new_item.expect in self.TERMINALS:
                        to_scan.add(new_item)
                    elif new_item not in column:
                        column.add(new_item)
                        items.append(new_item)

    def _parse(self, lexer, columns, to_scan, start_symbol=None):
        if False:
            i = 10
            return i + 15

        def is_quasi_complete(item):
            if False:
                return 10
            if item.is_complete:
                return True
            quasi = item.advance()
            while not quasi.is_complete:
                if quasi.expect not in self.NULLABLE:
                    return False
                if quasi.rule.origin == start_symbol and quasi.expect == start_symbol:
                    return False
                quasi = quasi.advance()
            return True

        def scan(i, token, to_scan):
            if False:
                print('Hello World!')
            'The core Earley Scanner.\n\n            This is a custom implementation of the scanner that uses the\n            Lark lexer to match tokens. The scan list is built by the\n            Earley predictor, based on the previously completed tokens.\n            This ensures that at each phase of the parse we have a custom\n            lexer context, allowing for more complex ambiguities.'
            next_to_scan = self.Set()
            next_set = self.Set()
            columns.append(next_set)
            transitives.append({})
            node_cache = {}
            for item in self.Set(to_scan):
                if match(item.expect, token):
                    new_item = item.advance()
                    label = (new_item.s, new_item.start, i)
                    term = terminals.get(token.type) if isinstance(token, Token) else None
                    token_node = TokenNode(token, term, priority=0)
                    new_item.node = node_cache[label] if label in node_cache else node_cache.setdefault(label, self.SymbolNode(*label))
                    new_item.node.add_family(new_item.s, item.rule, new_item.start, item.node, token_node)
                    if new_item.expect in self.TERMINALS:
                        next_to_scan.add(new_item)
                    else:
                        next_set.add(new_item)
            if not next_set and (not next_to_scan):
                expect = {i.expect.name for i in to_scan}
                raise UnexpectedToken(token, expect, considered_rules=set(to_scan), state=frozenset((i.s for i in to_scan)))
            return next_to_scan
        match = self.term_matcher
        terminals = self.lexer_conf.terminals_by_name
        transitives = [{}]
        expects = {i.expect for i in to_scan}
        i = 0
        for token in lexer.lex(expects):
            self.predict_and_complete(i, to_scan, columns, transitives)
            to_scan = scan(i, token, to_scan)
            i += 1
            expects.clear()
            expects |= {i.expect for i in to_scan}
        self.predict_and_complete(i, to_scan, columns, transitives)
        assert i == len(columns) - 1
        return to_scan

    def parse(self, lexer, start):
        if False:
            for i in range(10):
                print('nop')
        assert start, start
        start_symbol = NonTerminal(start)
        columns = [self.Set()]
        to_scan = self.Set()
        for rule in self.predictions[start_symbol]:
            item = Item(rule, 0, 0)
            if item.expect in self.TERMINALS:
                to_scan.add(item)
            else:
                columns[0].add(item)
        to_scan = self._parse(lexer, columns, to_scan, start_symbol)
        solutions = [n.node for n in columns[-1] if n.is_complete and n.node is not None and (n.s == start_symbol) and (n.start == 0)]
        if not solutions:
            expected_terminals = [t.expect.name for t in to_scan]
            raise UnexpectedEOF(expected_terminals, state=frozenset((i.s for i in to_scan)))
        if self.debug:
            from .earley_forest import ForestToPyDotVisitor
            try:
                debug_walker = ForestToPyDotVisitor()
            except ImportError:
                logger.warning("Cannot find dependency 'pydot', will not generate sppf debug image")
            else:
                debug_walker.visit(solutions[0], 'sppf.png')
        if len(solutions) > 1:
            assert False, 'Earley should not generate multiple start symbol items!'
        if self.Tree is not None:
            transformer = ForestToParseTree(self.Tree, self.callbacks, self.forest_sum_visitor and self.forest_sum_visitor(), self.resolve_ambiguity)
            return transformer.transform(solutions[0])
        return solutions[0]