from .downloader import BaseDownloader

class DeadDownloader(BaseDownloader):
    __name__ = 'DeadDownloader'
    __type__ = 'downloader'
    __version__ = '0.25'
    __status__ = 'stable'
    __pattern__ = '^unmatchable$'
    __config__ = [('enabled', 'bool', 'Activated', True)]
    __description__ = 'Downloader is no longer available'
    __license__ = 'GPLv3'
    __authors__ = [('zoidberg', 'zoidberg@mujmail.cz')]

    def get_info(self, *args, **kwargs):
        if False:
            print('Hello World!')
        info = super(DeadDownloader, self).get_info(*args, **kwargs)
        info['status'] = 1
        return info

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.offline(self._('Downloader is no longer available'))