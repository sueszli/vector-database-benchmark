from pynvim import Nvim
import re
import typing
from deoplete.base.filter import Base
from deoplete.util import UserContext, Candidates

class Filter(Base):

    def __init__(self, vim: Nvim) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(vim)
        self.name = 'converter_remove_overlap'
        self.description = 'remove overlap converter'

    def filter(self, context: UserContext) -> Candidates:
        if False:
            i = 10
            return i + 15
        if not context['next_input']:
            return list(context['candidates'])
        next_input_words = [x for x in re.split('([a-zA-Z_]+|\\W)', context['next_input']) if x]
        cur_pos = self.vim.call('getcurpos')[1:3]
        check_pairs = []
        pair_pos = self.vim.call('searchpairpos', '(', '', ')', 'nW')
        if '(' in context['input'] and cur_pos < pair_pos and (cur_pos[0] == pair_pos[0]):
            check_pairs.append(['(', ')', pair_pos])
        pair_pos = self.vim.call('searchpairpos', '[', '', ']', 'nW')
        if '[' in context['input'] and cur_pos < pair_pos and (cur_pos[0] == pair_pos[0]):
            check_pairs.append(['[', ']', pair_pos])
        for [overlap, candidate, word] in [[x, y, y['word']] for (x, y) in [[overlap_length(x['word'], next_input_words), x] for x in context['candidates']] if x > 0]:
            word_end_pos = context['complete_position'] + self.vim.call('len', word)
            if [x for x in check_pairs if x[0] in word and x[1] in word[-overlap:] and (word_end_pos >= x[2][1])]:
                continue
            if 'abbr' not in candidate:
                candidate['abbr'] = word
            candidate['word'] = word[:-overlap]
        return list(context['candidates'])

def overlap_length(left: str, next_input_words: typing.List[str]) -> int:
    if False:
        i = 10
        return i + 15
    pos = len(next_input_words)
    while pos > 0 and (not left.endswith(''.join(next_input_words[:pos]))):
        pos -= 1
    return len(''.join(next_input_words[:pos]))