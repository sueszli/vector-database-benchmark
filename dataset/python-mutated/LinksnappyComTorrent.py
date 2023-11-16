import json
import os
import time
import urllib.request
from pyload.core.network.http.http_request import FormFile
from pyload.core.utils import parse
from ..base.downloader import BaseDownloader
from ..helpers import exists

class LinksnappyComTorrent(BaseDownloader):
    __name__ = 'LinksnappyComTorrent'
    __type__ = 'downloader'
    __version__ = '0.01'
    __status__ = 'testing'
    __pattern__ = '^unmatchable$'
    __config__ = [('enabled', 'bool', 'Activated', True), ('folder_per_package', 'Default;Yes;No', 'Create folder for each package', 'Default'), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10), ('del_finished', 'bool', 'Delete downloaded torrents from the server', True)]
    __description__ = 'Linksnappy.com torrents decrypter plugin'
    __license__ = 'GPLv3'
    __authors__ = [('GammaC0de', 'nitzo2001[AT}yahoo[DOT]com')]
    API_URL = 'https://linksnappy.com/api/'

    def api_request(self, method, **kwargs):
        if False:
            i = 10
            return i + 15
        return json.loads(self.load(self.API_URL + method, get=kwargs))

    def sleep(self, sec):
        if False:
            for i in range(10):
                print('nop')
        for _i in range(sec):
            if self.pyfile.abort:
                break
            time.sleep(1)

    def send_request_to_server(self):
        if False:
            print('Hello World!')
        ' Send torrent/magnet to the server '
        if self.pyfile.url.endswith('.torrent'):
            if self.pyfile.url.startswith('http'):
                api_data = self.api_request('torrents/ADDURL', url=self.pyfile.url).items()[0][1]
                if api_data['status'] == 'FAILED' and api_data['error'] != 'This torrent already exists in your account':
                    self.fail(api_data['error'])
                torrent_id = api_data['torrentid']
            else:
                torrent_filename = urllib.request.url2pathname(self.pyfile.url[7:])
                if not exists(torrent_filename):
                    self.fail(self._('Torrent file does not exist'))
                if os.path.abspath(torrent_filename).startswith(self.pyload.tempdir + os.sep):
                    api_data = self.load('https://linksnappy.com/includes/ajaxupload.php', post={'torrents[]': FormFile(torrent_filename, mimetype='application/octet-stream')}, multipart=True)
                    api_data = list(json.loads(api_data).items())[0][1]
                    if api_data['error'] and api_data['error'] != 'This torrent already exists in your account':
                        self.fail(api_data['error'])
                    torrent_id = api_data['torrentid']
                else:
                    self.fail(self._('Illegal URL'))
        else:
            api_data = self.api_request('torrents/ADDMAGNET', magnetlinks=self.pyfile.url)
            if api_data['status'] != 'OK':
                self.fail(api_data['error'])
            api_data = api_data['return'][0]
            if api_data['status'] != 'OK' and api_data['error'] != 'This torrent already exists in your account':
                self.fail(api_data['error'])
            torrent_id = api_data['torrentid']
        return torrent_id

    def wait_for_server_dl(self, torrent_id):
        if False:
            print('Hello World!')
        ' Show progress while the server does the download '
        api_data = self.api_request('torrents/STATUS', tid=torrent_id)
        if api_data['status'] != 'OK':
            self.fail(api_data['error'])
        if api_data['return']['status'] == 'ERROR':
            self.fail(api_data['return']['error'])
        self.pyfile.name = api_data['return']['name']
        self.pyfile.set_custom_status('torrent')
        self.pyfile.set_progress(0)
        if api_data['return']['status'] != 'FINISHED':
            api_data = self.api_request('torrents/START', tid=torrent_id)
            if api_data['status'] != 'OK':
                if api_data['error'] == 'Magnet URI processing in progress. Please wait.':
                    for _i in range(8):
                        self.sleep(3)
                        api_data = self.api_request('torrents/START', tid=torrent_id)
                        if api_data['status'] == 'OK':
                            break
                    else:
                        self.fail(api_data['error'])
                elif api_data['error'] != 'Already started.':
                    self.fail(api_data['error'])
            while True:
                api_data = self.api_request('torrents/STATUS', tid=torrent_id)
                if api_data['status'] != 'OK':
                    self.fail(api_data['error'])
                if api_data['return']['status'] == 'ERROR':
                    self.fail(api_data['return']['error'])
                torrent_size = api_data['return'].get('getSize')
                if torrent_size is not None and self.pyfile.size == 0:
                    self.pyfile.size = parse.bytesize(torrent_size)
                progress = int(api_data['return']['percentDone'])
                self.pyfile.set_progress(progress)
                if api_data['return']['status'] == 'FINISHED':
                    break
                self.sleep(2)
        self.pyfile.set_progress(100)
        self.sleep(1)
        self.pyfile.set_custom_status('makezip')
        self.pyfile.set_progress(0)
        while True:
            api_data = self.api_request('torrents/GENZIP', torrentid=torrent_id)
            if api_data['status'] == 'ERROR':
                self.fail(api_data['error'])
            elif api_data['status'] == 'PENDING':
                self.sleep(2)
            else:
                break
        self.pyfile.set_progress(100)
        return api_data['return']

    def delete_torrent_from_server(self, torrent_id):
        if False:
            for i in range(10):
                print('nop')
        ' Remove the torrent from the server '
        self.api_request('torrents/DELETETORRENT', tid=torrent_id, delFiles=1)

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.multi_dl = True
        self.resume_download = True
        self.chunk_limit = 1
        if 'LinksnappyCom' not in self.pyload.account_manager.plugins:
            self.fail(self._('This plugin requires an active Linksnappy.com account'))
        self.account = self.pyload.account_manager.get_account_plugin('LinksnappyCom')
        if len(self.account.accounts) == 0:
            self.fail(self._('This plugin requires an active Linksnappy.com account'))
        self.load_account()
        cj = self.pyload.request_factory.get_cookie_jar('LinksnappyCom', self.account.user)
        self.req.set_cookie_jar(cj)

    def process(self, pyfile):
        if False:
            while True:
                i = 10
        torrent_id = False
        try:
            torrent_id = self.send_request_to_server()
            torrent_url = self.wait_for_server_dl(torrent_id)
            self.pyfile.name = os.path.basename(torrent_url)
            self.download(torrent_url)
        finally:
            if torrent_id is not False and self.config.get('del_finished'):
                self.delete_torrent_from_server(torrent_id)