from pynvim import Nvim
import re
from deoplete.base.filter import Base
from deoplete.util import UserContext, Candidates

class Filter(Base):

    def __init__(self, vim: Nvim) -> None:
        if False:
            while True:
                i = 10
        super().__init__(vim)
        self.name = 'converter_remove_paren'
        self.description = 'remove parentheses converter'

    def filter(self, context: UserContext) -> Candidates:
        if False:
            i = 10
            return i + 15
        for candidate in [x for x in context['candidates'] if '(' in x['word']]:
            candidate['word'] = re.sub('\\(.*\\)(\\$\\d+)?', '', candidate['word'])
        return list(context['candidates'])