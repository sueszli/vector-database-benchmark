import googlesearch
from .mb_get import MB_GET

class MB_GOOGLE(MB_GET):
    """
    This is a modified version of MB_GET.
    """

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        results = {}
        query = f'{self.observable_name} site:bazaar.abuse.ch'
        for url in googlesearch.search(query, stop=20):
            mb_hash = url.split('/')[-2]
            res = super().query_mb_api(observable_name=mb_hash)
            results[mb_hash] = res
        return results