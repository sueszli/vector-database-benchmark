from pynvim import Nvim
from deoplete.base.filter import Base
from deoplete.util import truncate_skipping, UserContext, Candidates

class Filter(Base):

    def __init__(self, vim: Nvim) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(vim)
        self.name = 'converter_truncate_menu'
        self.description = 'truncate menu converter'

    def filter(self, context: UserContext) -> Candidates:
        if False:
            print('Hello World!')
        max_width = context['max_menu_width']
        if not context['candidates'] or 'menu' not in context['candidates'][0] or max_width <= 0:
            return list(context['candidates'])
        footer_width = max_width / 3
        for candidate in context['candidates']:
            candidate['menu'] = truncate_skipping(candidate.get('menu', ''), max_width, '..', footer_width)
        return list(context['candidates'])