import requests.adapters

class BaseHTTPAdapter(requests.adapters.HTTPAdapter):

    def close(self):
        if False:
            i = 10
            return i + 15
        super().close()
        if hasattr(self, 'pools'):
            self.pools.clear()