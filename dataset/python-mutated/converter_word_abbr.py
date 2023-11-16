from pynvim import Nvim
from deoplete.base.filter import Base
from deoplete.util import UserContext, Candidates

class Filter(Base):

    def __init__(self, vim: Nvim) -> None:
        if False:
            print('Hello World!')
        super().__init__(vim)
        self.name = 'converter_word_abbr'
        self.description = 'word abbr converter'

    def filter(self, context: UserContext) -> Candidates:
        if False:
            i = 10
            return i + 15
        for candidate in context['candidates']:
            candidate['abbr'] = candidate['word']
        return list(context['candidates'])