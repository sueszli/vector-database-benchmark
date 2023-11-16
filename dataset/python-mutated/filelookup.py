import os.path
import re
from PyQt6 import QtCore
from picard import log
from picard.config import get_config
from picard.const import PICARD_URLS
from picard.disc import Disc
from picard.util import build_qurl, webbrowser2
from picard.ui.searchdialog.album import AlbumSearchDialog

class FileLookup(object):
    RE_MB_ENTITY = re.compile('\n        \\b(?P<entity>area|artist|instrument|label|place|recording|release|release-group|series|track|url|work)?\n        \\W*(?P<id>[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})\n    ', re.VERBOSE | re.IGNORECASE)
    RE_MB_CDTOC = re.compile('\n        \\b(?P<entity>cdtoc)\n        \\W*(?P<id>[a-z0-9-_.]{28})\n    ', re.VERBOSE | re.IGNORECASE)

    def __init__(self, parent, server, port, local_port):
        if False:
            i = 10
            return i + 15
        self.server = server
        self.local_port = int(local_port)
        self.port = port

    def _url(self, path, params=None):
        if False:
            while True:
                i = 10
        if params is None:
            params = {}
        if self.local_port:
            params['tport'] = self.local_port
        url = build_qurl(self.server, self.port, path=path, queryargs=params)
        return bytes(url.toEncoded()).decode()

    def _build_launch(self, path, params=None):
        if False:
            while True:
                i = 10
        if params is None:
            params = {}
        return self.launch(self._url(path, params))

    def launch(self, url):
        if False:
            i = 10
            return i + 15
        log.debug('webbrowser2: %s', url)
        webbrowser2.open(url)
        return True

    def _lookup(self, type_, id_):
        if False:
            i = 10
            return i + 15
        return self._build_launch('/%s/%s' % (type_, id_))

    def recording_lookup(self, recording_id):
        if False:
            while True:
                i = 10
        return self._lookup('recording', recording_id)

    def album_lookup(self, album_id):
        if False:
            i = 10
            return i + 15
        return self._lookup('release', album_id)

    def artist_lookup(self, artist_id):
        if False:
            return 10
        return self._lookup('artist', artist_id)

    def track_lookup(self, track_id):
        if False:
            print('Hello World!')
        return self._lookup('track', track_id)

    def work_lookup(self, work_id):
        if False:
            i = 10
            return i + 15
        return self._lookup('work', work_id)

    def release_group_lookup(self, release_group_id):
        if False:
            print('Hello World!')
        return self._lookup('release-group', release_group_id)

    def discid_lookup(self, discid):
        if False:
            while True:
                i = 10
        return self._lookup('cdtoc', discid)

    def discid_submission(self, url):
        if False:
            return 10
        if self.local_port:
            url = '%s&tport=%d' % (url, self.local_port)
        return self.launch(url)

    def acoust_lookup(self, acoust_id):
        if False:
            for i in range(10):
                print('nop')
        return self.launch(PICARD_URLS['acoustid_track'] + acoust_id)

    def mbid_lookup(self, string, type_=None, mbid_matched_callback=None, browser_fallback=True):
        if False:
            while True:
                i = 10
        "Parses string for known entity type and mbid, open browser for it\n        If entity type is 'release', it will load corresponding release if\n        possible.\n        "
        m = self.RE_MB_ENTITY.search(string)
        if m is None:
            m = self.RE_MB_CDTOC.search(string)
            if m is None:
                return False
        entity = m.group('entity')
        if entity is None:
            if type_ is None:
                return False
            entity = type_
        else:
            entity = entity.lower()
        id = m.group('id')
        if entity != 'cdtoc':
            id = id.lower()
        log.debug('Lookup for %s:%s', entity, id)
        if mbid_matched_callback:
            mbid_matched_callback(entity, id)
        if entity == 'release':
            QtCore.QObject.tagger.load_album(id)
            return True
        elif entity == 'recording':
            QtCore.QObject.tagger.load_nat(id)
            return True
        elif entity == 'release-group':
            AlbumSearchDialog.show_releasegroup_search(id)
            return True
        elif entity == 'cdtoc':
            disc = Disc(id=id)
            disc.lookup()
            return True
        if browser_fallback:
            return self._lookup(entity, id)
        return False

    def tag_lookup(self, artist, release, track, tracknum, duration, filename):
        if False:
            i = 10
            return i + 15
        params = {'artist': artist, 'release': release, 'track': track, 'tracknum': tracknum, 'duration': duration, 'filename': os.path.basename(filename)}
        return self._build_launch('/taglookup', params)

    def collection_lookup(self, userid):
        if False:
            print('Hello World!')
        return self._build_launch('/user/%s/collections' % userid)

    def search_entity(self, type_, query, adv=False, mbid_matched_callback=None, force_browser=False):
        if False:
            for i in range(10):
                print('nop')
        if not force_browser and self.mbid_lookup(query, type_, mbid_matched_callback=mbid_matched_callback):
            return True
        config = get_config()
        params = {'limit': config.setting['query_limit'], 'type': type_, 'query': query}
        if adv:
            params['adv'] = 'on'
        return self._build_launch('/search/textsearch', params)