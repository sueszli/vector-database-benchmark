from jadi import service
from aj.plugins.core.api.sidebar import SidebarItemProvider

@service
class Navbox:
    """
    Object to handle search queries through navbox.
    """

    def __init__(self, context):
        if False:
            for i in range(10):
                print('nop')
        self.context = context

    def search(self, query):
        if False:
            i = 10
            return i + 15
        query = query.strip().lower()
        results = []
        for provider in SidebarItemProvider.all(self.context):
            for item in provider.provide():
                if 'url' in item:
                    search_source = '$'.join([item.get('id', ''), item.get('name', '')]).lower()
                    if query in search_source:
                        results.append({'title': item['name'], 'icon': item['icon'], 'url': item['url']})
        return results