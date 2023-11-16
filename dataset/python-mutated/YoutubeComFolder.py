import json
from ..base.decrypter import BaseDecrypter

class YoutubeComFolder(BaseDecrypter):
    __name__ = 'YoutubeComFolder'
    __type__ = 'decrypter'
    __version__ = '1.12'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.|m\\.)?youtube\\.com/(?P<TYPE>user|playlist|view_play_list)(/|.*?[?&](?:list|p)=)(?P<ID>[\\w\\-]+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('folder_per_package', 'Default;Yes;No', 'Create folder for each package', 'Default'), ('likes', 'bool', 'Grab user (channel) liked videos', False), ('favorites', 'bool', 'Grab user (channel) favorite videos', False), ('uploads', 'bool', 'Grab channel unplaylisted videos', True)]
    __description__ = 'Youtube.com channel & playlist decrypter plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com')]
    API_KEY = 'AIzaSyB68u-qFPP9oBJpo1DWAPFE_VD2Sfy9hpk'

    def api_request(self, method, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['key'] = self.API_KEY
        json_data = self.load('https://www.googleapis.com/youtube/v3/' + method, get=kwargs)
        return json.loads(json_data)

    def get_channel(self, user):
        if False:
            print('Hello World!')
        channels = self.api_request('channels', part='id,snippet,contentDetails', forUsername=user, maxResults=50)
        if channels['items']:
            channel = channels['items'][0]
            return {'id': channel['id'], 'title': channel['snippet']['title'], 'relatedPlaylists': channel['contentDetails']['relatedPlaylists'], 'user': user}

    def get_playlist(self, playlist_id):
        if False:
            while True:
                i = 10
        playlists = self.api_request('playlists', part='snippet', id=playlist_id)
        if playlists['items']:
            playlist = playlists['items'][0]
            return {'id': playlist_id, 'title': playlist['snippet']['title'], 'channelId': playlist['snippet']['channelId'], 'channelTitle': playlist['snippet']['channelTitle']}

    def _get_playlists(self, playlist_id, token=None):
        if False:
            return 10
        if token:
            playlists = self.api_request('playlists', part='id', maxResults=50, channelId=playlist_id, pageToken=token)
        else:
            playlists = self.api_request('playlists', part='id', maxResults=50, channelId=playlist_id)
        for playlist in playlists['items']:
            yield playlist['id']
        if 'nextPageToken' in playlists:
            for item in self._get_playlists(playlist_id, playlists['nextPageToken']):
                yield item

    def get_playlists(self, ch_id):
        if False:
            i = 10
            return i + 15
        return [self.get_playlist(p_id) for p_id in self._get_playlists(ch_id)]

    def _get_videos_id(self, playlist_id, token=None):
        if False:
            return 10
        if token:
            playlist = self.api_request('playlistItems', part='contentDetails', maxResults=50, playlistId=playlist_id, pageToken=token)
        else:
            playlist = self.api_request('playlistItems', part='contentDetails', maxResults=50, playlistId=playlist_id)
        for item in playlist['items']:
            yield item['contentDetails']['videoId']
        if 'nextPageToken' in playlist:
            for item in self._get_videos_id(playlist_id, playlist['nextPageToken']):
                yield item

    def get_videos_id(self, p_id):
        if False:
            return 10
        return list(self._get_videos_id(p_id))

    def decrypt(self, pyfile):
        if False:
            for i in range(10):
                print('nop')
        if self.info['pattern']['TYPE'] == 'user':
            self.log_debug('Url recognized as Channel')
            channel = self.get_channel(self.info['pattern']['ID'])
            if channel:
                playlists = self.get_playlists(channel['id'])
                self.log_debug('{} playlists found on channel "{}"'.format(len(playlists), channel['title']))
                related_playlist = {p_name: self.get_playlist(p_id) for (p_name, p_id) in channel['relatedPlaylists'].items()}
                self.log_debug("Channel's related playlists found = {}".format(list(related_playlist.keys())))
                related_playlist['uploads']['title'] = 'Unplaylisted videos'
                related_playlist['uploads']['checkDups'] = True
                for (p_name, p_data) in related_playlist.items():
                    if self.config.get(p_name):
                        p_data['title'] += ' of ' + channel['user']
                        playlists.append(p_data)
            else:
                playlists = []
        else:
            self.log_debug('Url recognized as Playlist')
            playlists = [self.get_playlist(self.info['pattern']['ID'])]
        if not playlists:
            self.fail(self._('No playlist available'))
        added_videos = []
        urlize = lambda x: 'https://www.youtube.com/watch?v=' + x
        for p in playlists:
            p_name = p['title']
            p_videos = self.get_videos_id(p['id'])
            self.log_debug('{} videos found on playlist "{}"'.format(len(p_videos), p_name))
            if not p_videos:
                continue
            elif 'checkDups' in p:
                p_urls = [urlize(v_id) for v_id in p_videos if v_id not in added_videos]
                self.log_debug('{} videos available on playlist "{}" after duplicates cleanup'.format(len(p_urls), p_name))
            else:
                p_urls = [urlize(url) for url in p_videos]
            self.packages.append((p_name, p_urls, p_name))
            added_videos.extend(p_videos)