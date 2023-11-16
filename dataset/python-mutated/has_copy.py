from awxkit.api.pages import Page
from awxkit.utils import random_title

class HasCopy(object):

    def can_copy(self):
        if False:
            while True:
                i = 10
        return self.get_related('copy').can_copy

    def copy(self, name=''):
        if False:
            i = 10
            return i + 15
        'Return a copy of current page'
        payload = {'name': name or 'Copy - ' + random_title()}
        endpoint = self.json.related['copy']
        page = Page(self.connection, endpoint=endpoint)
        return page.post(payload)