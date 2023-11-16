from pynvim import Nvim
from deoplete.base.filter import Base
from deoplete.util import UserContext, Candidates

class Filter(Base):

    def __init__(self, vim: Nvim) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(vim)
        self.name = 'sorter_word'
        self.description = 'word sorter'

    def filter(self, context: UserContext) -> Candidates:
        if False:
            i = 10
            return i + 15
        return sorted(context['candidates'], key=lambda x: str(x['word'].swapcase()))