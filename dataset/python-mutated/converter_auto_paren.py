from pynvim import Nvim
import re
from deoplete.base.filter import Base
from deoplete.util import UserContext, Candidates

class Filter(Base):

    def __init__(self, vim: Nvim) -> None:
        if False:
            return 10
        super().__init__(vim)
        self.name = 'converter_auto_paren'
        self.description = 'auto add parentheses converter'

    def filter(self, context: UserContext) -> Candidates:
        if False:
            return 10
        p1 = re.compile('\\(\\)?$')
        p2 = re.compile('\\(.*\\)')
        for candidate in [x for x in context['candidates'] if not p1.search(x['word']) and ('abbr' in x and p2.search(x['abbr']) or ('info' in x and p2.search(x['info'])))]:
            candidate['word'] += '('
        return list(context['candidates'])