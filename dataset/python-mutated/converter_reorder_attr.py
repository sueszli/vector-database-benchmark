from pynvim import Nvim
import re
import typing
from deoplete.base.filter import Base
from deoplete.util import UserContext, Candidates

class Filter(Base):

    def __init__(self, vim: Nvim) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(vim)
        self.name = 'converter_reorder_attr'
        self.description = 'Reorder candidates based on their attributes'
        self.vars = {'attrs_order': {}}

    @staticmethod
    def filter_attrs(candidates: Candidates, preferred_order_attrs: typing.Dict[str, typing.Any], max_list_size: int=500) -> Candidates:
        if False:
            print('Hello World!')
        context_candidates = candidates[:]
        new_candidates = []
        new_candidates_len = 0
        for attr in preferred_order_attrs.keys():
            for expr in preferred_order_attrs[attr]:
                disabled = expr[0] == '!'
                if disabled:
                    expr = expr[1:]
                expr = re.compile(expr)
                size = len(context_candidates)
                i = 0
                while i < size:
                    candidate = context_candidates[i]
                    if attr in candidate and expr.search(candidate[attr]):
                        candidate = context_candidates.pop(i)
                        i -= 1
                        size -= 1
                        if not disabled:
                            new_candidates.append(candidate)
                            new_candidates_len += 1
                            if new_candidates_len == max_list_size:
                                return new_candidates
                    i += 1
            new_candidates.extend(context_candidates)
            context_candidates = new_candidates
        return new_candidates

    def filter(self, context: UserContext) -> Candidates:
        if False:
            print('Hello World!')
        preferred_order_attrs = self.get_var('attrs_order').get(context['filetype'], [])
        if not context['candidates'] or not preferred_order_attrs:
            return list(context['candidates'])
        max_list_size = self.vim.call('deoplete#custom#_get_option', 'max_list')
        return self.filter_attrs(context['candidates'], preferred_order_attrs, max_list_size)