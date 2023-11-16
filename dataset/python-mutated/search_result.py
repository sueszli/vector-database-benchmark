__license__ = 'GPL 3'
__copyright__ = '2011, John Schember <john@nachtimwald.com>'
__docformat__ = 'restructuredtext en'

class SearchResult:
    DRM_LOCKED = 1
    DRM_UNLOCKED = 2
    DRM_UNKNOWN = 3

    def __init__(self):
        if False:
            while True:
                i = 10
        self.store_name = ''
        self.cover_url = ''
        self.cover_data = None
        self.title = ''
        self.author = ''
        self.price = ''
        self.detail_item = ''
        self.drm = None
        self.formats = ''
        self.downloads = {}
        self.affiliate = False
        self.plugin_author = ''
        self.create_browser = None

    def __eq__(self, other):
        if False:
            return 10
        return self.title == other.title and self.author == other.author and (self.store_name == other.store_name) and (self.formats == other.formats)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.title, self.author, self.store_name, self.formats))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        items = []
        for x in 'store_name title author price formats detail_item cover_url'.split():
            items.append(f'\t{x}={getattr(self, x)!r}')
        return 'SearchResult(\n%s\n)' % '\n'.join(items)
    __repr__ = __str__
    __unicode__ = __str__