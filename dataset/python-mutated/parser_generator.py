import contextlib
from abc import abstractmethod
from typing import AbstractSet, Dict, IO, Iterator, List, Optional, Set, Text, Tuple
from pegen import sccutils
from pegen.grammar import Grammar, Rule, Rhs, Alt, NamedItem, Plain, NameLeaf, Gather
from pegen.grammar import GrammarError, GrammarVisitor

class RuleCheckingVisitor(GrammarVisitor):

    def __init__(self, rules: Dict[str, Rule], tokens: Dict[int, str]):
        if False:
            return 10
        self.rules = rules
        self.tokens = tokens

    def visit_NameLeaf(self, node: NameLeaf) -> None:
        if False:
            while True:
                i = 10
        if node.value not in self.rules and node.value not in self.tokens.values():
            raise GrammarError(f'Dangling reference to rule {node.value!r}')

    def visit_NamedItem(self, node: NamedItem) -> None:
        if False:
            while True:
                i = 10
        if node.name and node.name.startswith('_'):
            raise GrammarError(f"Variable names cannot start with underscore: '{node.name}'")
        self.visit(node.item)

class ParserGenerator:
    callmakervisitor: GrammarVisitor

    def __init__(self, grammar: Grammar, tokens: Dict[int, str], file: Optional[IO[Text]]):
        if False:
            while True:
                i = 10
        self.grammar = grammar
        self.tokens = tokens
        self.rules = grammar.rules
        self.validate_rule_names()
        if 'trailer' not in grammar.metas and 'start' not in self.rules:
            raise GrammarError("Grammar without a trailer must have a 'start' rule")
        checker = RuleCheckingVisitor(self.rules, self.tokens)
        for rule in self.rules.values():
            checker.visit(rule)
        self.file = file
        self.level = 0
        compute_nullables(self.rules)
        (self.first_graph, self.first_sccs) = compute_left_recursives(self.rules)
        self.todo = self.rules.copy()
        self.counter = 0
        self.keyword_counter = 499
        self.all_rules: Dict[str, Rule] = {}
        self._local_variable_stack: List[List[str]] = []

    def validate_rule_names(self) -> None:
        if False:
            while True:
                i = 10
        for rule in self.rules:
            if rule.startswith('_'):
                raise GrammarError(f"Rule names cannot start with underscore: '{rule}'")

    @contextlib.contextmanager
    def local_variable_context(self) -> Iterator[None]:
        if False:
            for i in range(10):
                print('nop')
        self._local_variable_stack.append([])
        yield
        self._local_variable_stack.pop()

    @property
    def local_variable_names(self) -> List[str]:
        if False:
            return 10
        return self._local_variable_stack[-1]

    @abstractmethod
    def generate(self, filename: str) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @contextlib.contextmanager
    def indent(self) -> Iterator[None]:
        if False:
            i = 10
            return i + 15
        self.level += 1
        try:
            yield
        finally:
            self.level -= 1

    def print(self, *args: object) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not args:
            print(file=self.file)
        else:
            print('    ' * self.level, end='', file=self.file)
            print(*args, file=self.file)

    def printblock(self, lines: str) -> None:
        if False:
            i = 10
            return i + 15
        for line in lines.splitlines():
            self.print(line)

    def collect_todo(self) -> None:
        if False:
            return 10
        done: Set[str] = set()
        while True:
            alltodo = list(self.todo)
            self.all_rules.update(self.todo)
            todo = [i for i in alltodo if i not in done]
            if not todo:
                break
            for rulename in todo:
                self.todo[rulename].collect_todo(self)
            done = set(alltodo)

    def keyword_type(self) -> int:
        if False:
            return 10
        self.keyword_counter += 1
        return self.keyword_counter

    def name_node(self, rhs: Rhs) -> str:
        if False:
            while True:
                i = 10
        self.counter += 1
        name = f'_tmp_{self.counter}'
        self.todo[name] = Rule(name, None, rhs)
        return name

    def name_loop(self, node: Plain, is_repeat1: bool) -> str:
        if False:
            print('Hello World!')
        self.counter += 1
        if is_repeat1:
            prefix = '_loop1_'
        else:
            prefix = '_loop0_'
        name = f'{prefix}{self.counter}'
        self.todo[name] = Rule(name, None, Rhs([Alt([NamedItem(None, node)])]))
        return name

    def name_gather(self, node: Gather) -> str:
        if False:
            while True:
                i = 10
        self.counter += 1
        name = f'_gather_{self.counter}'
        self.counter += 1
        extra_function_name = f'_loop0_{self.counter}'
        extra_function_alt = Alt([NamedItem(None, node.separator), NamedItem('elem', node.node)], action='elem')
        self.todo[extra_function_name] = Rule(extra_function_name, None, Rhs([extra_function_alt]))
        alt = Alt([NamedItem('elem', node.node), NamedItem('seq', NameLeaf(extra_function_name))])
        self.todo[name] = Rule(name, None, Rhs([alt]))
        return name

    def dedupe(self, name: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        origname = name
        counter = 0
        while name in self.local_variable_names:
            counter += 1
            name = f'{origname}_{counter}'
        self.local_variable_names.append(name)
        return name

def compute_nullables(rules: Dict[str, Rule]) -> None:
    if False:
        return 10
    'Compute which rules in a grammar are nullable.\n\n    Thanks to TatSu (tatsu/leftrec.py) for inspiration.\n    '
    for rule in rules.values():
        rule.nullable_visit(rules)

def compute_left_recursives(rules: Dict[str, Rule]) -> Tuple[Dict[str, AbstractSet[str]], List[AbstractSet[str]]]:
    if False:
        i = 10
        return i + 15
    graph = make_first_graph(rules)
    sccs = list(sccutils.strongly_connected_components(graph.keys(), graph))
    for scc in sccs:
        if len(scc) > 1:
            for name in scc:
                rules[name].left_recursive = True
            leaders = set(scc)
            for start in scc:
                for cycle in sccutils.find_cycles_in_scc(graph, scc, start):
                    leaders -= scc - set(cycle)
                    if not leaders:
                        raise ValueError(f'SCC {scc} has no leadership candidate (no element is included in all cycles)')
            leader = min(leaders)
            rules[leader].leader = True
        else:
            name = min(scc)
            if name in graph[name]:
                rules[name].left_recursive = True
                rules[name].leader = True
    return (graph, sccs)

def make_first_graph(rules: Dict[str, Rule]) -> Dict[str, AbstractSet[str]]:
    if False:
        i = 10
        return i + 15
    "Compute the graph of left-invocations.\n\n    There's an edge from A to B if A may invoke B at its initial\n    position.\n\n    Note that this requires the nullable flags to have been computed.\n    "
    graph = {}
    vertices: Set[str] = set()
    for (rulename, rhs) in rules.items():
        graph[rulename] = names = rhs.initial_names()
        vertices |= names
    for vertex in vertices:
        graph.setdefault(vertex, set())
    return graph