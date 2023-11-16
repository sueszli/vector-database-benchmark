from pynvim import Nvim
from deoplete.base.filter import Base
from deoplete.util import UserContext, Candidates

class Filter(Base):

    def __init__(self, vim: Nvim) -> None:
        if False:
            print('Hello World!')
        super().__init__(vim)
        self.name = 'matcher_length'
        self.description = 'length matcher'

    def filter(self, context: UserContext) -> Candidates:
        if False:
            i = 10
            return i + 15
        input_len = len(context['complete_str'])
        return [x for x in context['candidates'] if len(x['word']) > input_len]