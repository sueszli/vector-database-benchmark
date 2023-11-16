from pynvim import Nvim
from deoplete.base.filter import Base
from deoplete.util import truncate_skipping, UserContext, Candidates

class Filter(Base):

    def __init__(self, vim: Nvim) -> None:
        if False:
            while True:
                i = 10
        super().__init__(vim)
        self.name = 'converter_truncate_info'
        self.description = 'truncate info converter'

    def filter(self, context: UserContext) -> Candidates:
        if False:
            while True:
                i = 10
        max_width = context['max_info_width']
        if not context['candidates'] or max_width <= 0:
            return list(context['candidates'])
        footer_width = 1
        for candidate in context['candidates']:
            candidate['info'] = truncate_skipping(candidate.get('info', ''), max_width, '..', footer_width)
        return list(context['candidates'])