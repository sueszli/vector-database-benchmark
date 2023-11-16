import json
from ..base.multi_downloader import MultiDownloader

class SmoozedCom(MultiDownloader):
    __name__ = 'SmoozedCom'
    __type__ = 'downloader'
    __version__ = '0.15'
    __status__ = 'testing'
    __pattern__ = '^unmatchable$'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', False), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10), ('revert_failed', 'bool', 'Revert to standard download if fails', True)]
    __description__ = 'Smoozed.com downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [(None, None)]
    FILE_ERRORS = [('Error', b'{"state":"error"}'), ('Retry', b'{"state":"retry"}')]

    def handle_free(self, pyfile):
        if False:
            i = 10
            return i + 15
        pyfile.name = pyfile.name.split('/').pop()
        suffix_to_remove = ['html', 'htm', 'php', 'php3', 'asp', 'shtm', 'shtml', 'cfml', 'cfm']
        temp = pyfile.name.split('.')
        if temp.pop() in suffix_to_remove:
            pyfile.name = '.'.join(temp)
        get_data = {'session_key': self.account.get_data('session'), 'url': pyfile.url}
        html = self.load('http://www2.smoozed.com/api/check', get=get_data)
        data = json.loads(html)
        if data['state'] != 'ok':
            self.fail(data['message'])
        if data['data'].get('state', 'ok') != 'ok':
            if data['data'] == 'Offline':
                self.offline()
            else:
                self.fail(data['data']['message'])
        pyfile.name = data['data']['name']
        pyfile.size = int(data['data']['size'])
        header = self.load('http://www2.smoozed.com/api/download', get=get_data, just_header=True)
        if 'location' not in header:
            self.fail(self._('Unable to initialize download'))
        else:
            self.link = header.get('location')[-1] if isinstance(header.get('location'), list) else header.get('location')