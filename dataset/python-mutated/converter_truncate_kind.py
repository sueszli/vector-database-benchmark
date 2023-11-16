from pynvim import Nvim
from deoplete.base.filter import Base
from deoplete.util import truncate_skipping, UserContext, Candidates

class Filter(Base):

    def __init__(self, vim: Nvim) -> None:
        if False:
            while True:
                i = 10
        super().__init__(vim)
        self.name = 'converter_truncate_kind'
        self.description = 'truncate kind converter'

    def filter(self, context: UserContext) -> Candidates:
        if False:
            i = 10
            return i + 15
        max_width = context['max_kind_width']
        if not context['candidates'] or 'kind' not in context['candidates'][0] or max_width <= 0:
            return list(context['candidates'])
        footer_width = max_width / 3
        for candidate in context['candidates']:
            candidate['kind'] = truncate_skipping(candidate.get('kind', ''), max_width, '..', footer_width)
        return list(context['candidates'])