"""Fixer that changes filter(F, X) into list(filter(F, X)).

We avoid the transformation if the filter() call is directly contained
in iter(<>), list(<>), tuple(<>), sorted(<>), ...join(<>), or
for V in <>:.

NOTE: This is still not correct if the original code was depending on
filter(F, X) to return a string if X is a string and a tuple if X is a
tuple.  That would require type inference, which we don't do.  Let
Python 2.6 figure it out.
"""
from .. import fixer_base
from ..pytree import Node
from ..pygram import python_symbols as syms
from ..fixer_util import Name, ArgList, ListComp, in_special_context, parenthesize

class FixFilter(fixer_base.ConditionalFix):
    BM_compatible = True
    PATTERN = "\n    filter_lambda=power<\n        'filter'\n        trailer<\n            '('\n            arglist<\n                lambdef< 'lambda'\n                         (fp=NAME | vfpdef< '(' fp=NAME ')'> ) ':' xp=any\n                >\n                ','\n                it=any\n            >\n            ')'\n        >\n        [extra_trailers=trailer*]\n    >\n    |\n    power<\n        'filter'\n        trailer< '(' arglist< none='None' ',' seq=any > ')' >\n        [extra_trailers=trailer*]\n    >\n    |\n    power<\n        'filter'\n        args=trailer< '(' [any] ')' >\n        [extra_trailers=trailer*]\n    >\n    "
    skip_on = 'future_builtins.filter'

    def transform(self, node, results):
        if False:
            i = 10
            return i + 15
        if self.should_skip(node):
            return
        trailers = []
        if 'extra_trailers' in results:
            for t in results['extra_trailers']:
                trailers.append(t.clone())
        if 'filter_lambda' in results:
            xp = results.get('xp').clone()
            if xp.type == syms.test:
                xp.prefix = ''
                xp = parenthesize(xp)
            new = ListComp(results.get('fp').clone(), results.get('fp').clone(), results.get('it').clone(), xp)
            new = Node(syms.power, [new] + trailers, prefix='')
        elif 'none' in results:
            new = ListComp(Name('_f'), Name('_f'), results['seq'].clone(), Name('_f'))
            new = Node(syms.power, [new] + trailers, prefix='')
        else:
            if in_special_context(node):
                return None
            args = results['args'].clone()
            new = Node(syms.power, [Name('filter'), args], prefix='')
            new = Node(syms.power, [Name('list'), ArgList([new])] + trailers)
            new.prefix = ''
        new.prefix = node.prefix
        return new