"""Processors for engine-type: ``online_url_search``

"""
import re
from .online import OnlineProcessor
re_search_urls = {'http': re.compile('https?:\\/\\/[^ ]*'), 'ftp': re.compile('ftps?:\\/\\/[^ ]*'), 'data:image': re.compile('data:image/[^; ]*;base64,[^ ]*')}

class OnlineUrlSearchProcessor(OnlineProcessor):
    """Processor class used by ``online_url_search`` engines."""
    engine_type = 'online_url_search'

    def get_params(self, search_query, engine_category):
        if False:
            return 10
        'Returns a set of :ref:`request params <engine request online>` or ``None`` if\n        search query does not match to :py:obj:`re_search_urls`.\n        '
        params = super().get_params(search_query, engine_category)
        if params is None:
            return None
        url_match = False
        search_urls = {}
        for (k, v) in re_search_urls.items():
            m = v.search(search_query.query)
            v = None
            if m:
                url_match = True
                v = m[0]
            search_urls[k] = v
        if not url_match:
            return None
        params['search_urls'] = search_urls
        return params