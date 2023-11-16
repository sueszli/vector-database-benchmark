class DnsCache:
    """
	The DnsCache maintains a cache of DNS lookups, mirroring the browser experience.
	"""
    _instance = None

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.customAddress = None
        self.cache = {}

    @staticmethod
    def getInstance():
        if False:
            i = 10
            return i + 15
        if DnsCache._instance == None:
            DnsCache._instance = DnsCache()
        return DnsCache._instance

    def cacheResolution(self, host, address):
        if False:
            return 10
        self.cache[host] = address

    def getCachedAddress(self, host):
        if False:
            i = 10
            return i + 15
        if host in self.cache:
            return self.cache[host]
        return None