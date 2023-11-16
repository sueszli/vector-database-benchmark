from ..base.simple_decrypter import SimpleDecrypter

class FilefactoryComFolder(SimpleDecrypter):
    __name__ = 'FilefactoryComFolder'
    __type__ = 'decrypter'
    __version__ = '0.38'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?filefactory\\.com/(?:f|folder)/\\w+'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('folder_per_package', 'Default;Yes;No', 'Create folder for each package', 'Default'), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Filefactory.com folder decrypter plugin'
    __license__ = 'GPLv3'
    __authors__ = [('stickell', 'l.stickell@yahoo.it')]
    COOKIES = [('filefactory.com', 'locale', 'en_US.utf-8')]
    LINK_PATTERN = '<td>\\s*<a href="(.+?)"'
    NAME_PATTERN = '<h1>Files in <span>(?P<N>.+?)<'
    PAGES_PATTERN = 'data-paginator-totalPages="(\\d+)'

    def load_page(self, page_n):
        if False:
            while True:
                i = 10
        return self.load(self.pyfile.url, get={'page': page_n, 'show': 100})