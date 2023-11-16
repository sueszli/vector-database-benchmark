from .decrypter import BaseDecrypter

class DeadDecrypter(BaseDecrypter):
    __name__ = 'DeadDecrypter'
    __type__ = 'decrypter'
    __version__ = '0.15'
    __status__ = 'stable'
    __pattern__ = '^unmatchable$'
    __config__ = [('enabled', 'bool', 'Activated', True)]
    __description__ = 'Decrypter is no longer available'
    __license__ = 'GPLv3'
    __authors__ = [('stickell', 'l.stickell@yahoo.it')]

    def get_info(self, *args, **kwargs):
        if False:
            return 10
        info = super(DeadDecrypter, self).get_info(*args, **kwargs)
        info['status'] = 1
        return info

    def setup(self):
        if False:
            return 10
        self.offline(self._('Decrypter is no longer available'))