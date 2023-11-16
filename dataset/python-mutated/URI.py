class URI(object):
    DEFAULTS = {'scheme': 'http', 'username': 'MYUSERNAME', 'password': 'MYPASSWORD', 'host': 'host.com'}

    def __init__(self, description='N/A', scheme=DEFAULTS['scheme'], username=DEFAULTS['username'], password=DEFAULTS['password'], host=DEFAULTS['host']):
        if False:
            return 10
        self.description = description
        self.scheme = scheme
        self.username = username
        self.password = password
        self.host = host

    def get_uri(self):
        if False:
            i = 10
            return i + 15
        uri = '%s://' % self.scheme
        if self.username:
            uri += '%s' % self.username
        if self.password:
            uri += ':%s' % self.password
        if (self.username or self.password) and self.host is not None:
            uri += '@%s' % self.host
        elif self.host is not None:
            uri += '%s' % self.host
        return uri

    def get_secret_count(self):
        if False:
            while True:
                i = 10
        secret_count = 0
        if self.username:
            secret_count += 1
        if self.password:
            secret_count += 1
        return secret_count

    def __string__(self):
        if False:
            print('Hello World!')
        return self.get_uri()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.get_uri()