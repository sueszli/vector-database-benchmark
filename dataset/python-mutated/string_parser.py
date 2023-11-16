from functools import partial
import re
from typing import Callable, Dict, List
from rocketry.pybox.string.parse import ClosureParser
from rocketry.pybox.container.visitor import Visitor

class InstructionParser:

    def __init__(self, item_parser: Callable, operators: List[Dict[str, Callable]]):
        if False:
            for i in range(10):
                print('nop')
        self.item_parser = item_parser
        self.operators = operators
        self.symbols = set((oper['symbol'] for oper in operators))

    def __call__(self, s: str, **kwargs):
        if False:
            while True:
                i = 10
        'Parse a string to condition. Allows logical operators.\n\n        Reserved keywords:\n            "&" : and operator\n            "|" : or operator\n            "~" : not operator\n            "(" : opening closure\n            ")" : closing closure\n\n        These characters cannot be found in\n        individual condition parsing (ie.\n        in the names of tasks).\n        '
        p = ClosureParser()
        v = Visitor(visit_types=(list,))
        l = p.to_list(s)
        v.assign_elements(l, self._split_operations)
        v.apply(l, _flatten_tuples)
        v.assign_elements(l, partial(self._parse, **kwargs))
        e = v.reduce(l, self._assemble)
        return e

    def _parse(self, __s: tuple, **kwargs):
        if False:
            while True:
                i = 10
        s = __s

        def parse_string(s):
            if False:
                return 10
            s = s.strip()
            if s in ('&', '|', '~'):
                return s
            return self.item_parser(s, **kwargs)
        if isinstance(s, str):
            return parse_string(s)
        return tuple((parse_string(e) for e in s))

    def _assemble(self, *s: tuple):
        if False:
            while True:
                i = 10
        v = Visitor(visit_types=(list, tuple))
        s = v.flatten(s)
        for operator in self.operators:
            oper_str = operator['symbol']
            oper_func = operator['func']
            oper_side = operator['side']
            s = self._assemble_oper(s, oper_str=oper_str, oper_func=oper_func, side=oper_side)
        return s[0] if isinstance(s, tuple) else s

    def _assemble_oper(self, s: list, oper_str: str, oper_func: Callable, side='both'):
        if False:
            while True:
                i = 10
        s = list(reversed(s))
        while self._contains_operator(s, oper_str):
            pos = self._index(s, [oper_str])
            if side == 'both':
                obj = oper_func(s[pos + 1], s[pos - 1])
                s[pos] = obj
                del s[pos - 1]
                del s[pos + 1 - 1]
            elif side == 'right':
                obj = oper_func(s[pos - 1])
                s[pos - 1] = obj
                del s[pos]
            elif side == 'left':
                obj = oper_func(s[pos + 1])
                s[pos + 1] = obj
                del s[pos]
        return tuple(reversed(s))

    @staticmethod
    def _contains_operator(s: list, oper_str: str):
        if False:
            print('Hello World!')
        for e in s:
            if isinstance(e, str) and e in (oper_str,):
                return True
        return False

    @staticmethod
    def _index(s: list, items: list):
        if False:
            print('Hello World!')
        'Get '
        for (i, e) in enumerate(s):
            if isinstance(e, str) and e in items:
                return i
        raise KeyError

    def _split_operations(self, s: str):
        if False:
            i = 10
            return i + 15
        symbols = ''.join(self.symbols)
        regex = '([' + symbols + '])'
        s = s.strip()
        if bool(re.search(regex, s)):
            l = re.split(regex, s)
            l = [elem for elem in l if elem.strip()]
            if len(l) == 1:
                return l[0]
            return tuple(l)
        return s

def _flatten_tuples(cont):
    if False:
        for i in range(10):
            print('nop')
    for (i, item) in enumerate(cont):
        if isinstance(item, tuple):
            cont.pop(i)
            for (j, tpl_item) in enumerate(item):
                cont.insert(i + j, tpl_item)
    return cont