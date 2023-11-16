class WihRecord:

    def __init__(self, record_type, content, source, site, fnv_hash):
        if False:
            i = 10
            return i + 15
        self.recordType = record_type
        self.content = content
        self.source = source
        self.site = site
        self.fnv_hash = fnv_hash

    def __str__(self):
        if False:
            print('Hello World!')
        return '{} {} {} {}'.format(self.recordType, self.content, self.source, self.site)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<WihRecord>' + self.__str__()

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.fnv_hash == other.fnv_hash

    def __hash__(self):
        if False:
            while True:
                i = 10
        return self.fnv_hash

    def dump_json(self):
        if False:
            return 10
        return {'record_type': self.recordType, 'content': self.content, 'site': self.site, 'source': self.source, 'fnv_hash': str(self.fnv_hash)}