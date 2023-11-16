import logging
from module.conf import settings
from module.models import Bangumi, Torrent
from module.network import RequestContent
from .path import TorrentPath
logger = logging.getLogger(__name__)

class DownloadClient(TorrentPath):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.client = self.__getClient()
        self.authed = False

    @staticmethod
    def __getClient():
        if False:
            i = 10
            return i + 15
        type = settings.downloader.type
        host = settings.downloader.host
        username = settings.downloader.username
        password = settings.downloader.password
        ssl = settings.downloader.ssl
        if type == 'qbittorrent':
            from .client.qb_downloader import QbDownloader
            return QbDownloader(host, username, password, ssl)
        else:
            logger.error(f'[Downloader] Unsupported downloader type: {type}')
            raise Exception(f'Unsupported downloader type: {type}')

    def __enter__(self):
        if False:
            while True:
                i = 10
        if not self.authed:
            self.auth()
        else:
            logger.error('[Downloader] Already authed.')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        if self.authed:
            self.client.logout()
            self.authed = False

    def auth(self):
        if False:
            print('Hello World!')
        self.authed = self.client.auth()
        if self.authed:
            logger.debug('[Downloader] Authed.')
        else:
            logger.error('[Downloader] Auth failed.')

    def check_host(self):
        if False:
            while True:
                i = 10
        return self.client.check_host()

    def init_downloader(self):
        if False:
            return 10
        prefs = {'rss_auto_downloading_enabled': True, 'rss_max_articles_per_feed': 500, 'rss_processing_enabled': True, 'rss_refresh_interval': 30}
        self.client.prefs_init(prefs=prefs)
        try:
            self.client.add_category('BangumiCollection')
        except Exception:
            logger.debug('[Downloader] Cannot add new category, maybe already exists.')
        if settings.downloader.path == '':
            prefs = self.client.get_app_prefs()
            settings.downloader.path = self._join_path(prefs['save_path'], 'Bangumi')

    def set_rule(self, data: Bangumi):
        if False:
            i = 10
            return i + 15
        data.rule_name = self._rule_name(data)
        data.save_path = self._gen_save_path(data)
        rule = {'enable': True, 'mustContain': data.title_raw, 'mustNotContain': '|'.join(data.filter), 'useRegex': True, 'episodeFilter': '', 'smartFilter': False, 'previouslyMatchedEpisodes': [], 'affectedFeeds': data.rss_link, 'ignoreDays': 0, 'lastMatch': '', 'addPaused': False, 'assignedCategory': 'Bangumi', 'savePath': data.save_path}
        self.client.rss_set_rule(rule_name=data.rule_name, rule_def=rule)
        data.added = True
        logger.info(f'[Downloader] Add {data.official_title} Season {data.season} to auto download rules.')

    def set_rules(self, bangumi_info: list[Bangumi]):
        if False:
            print('Hello World!')
        logger.debug('[Downloader] Start adding rules.')
        for info in bangumi_info:
            self.set_rule(info)
        logger.debug('[Downloader] Finished.')

    def get_torrent_info(self, category='Bangumi', status_filter='completed', tag=None):
        if False:
            for i in range(10):
                print('nop')
        return self.client.torrents_info(status_filter=status_filter, category=category, tag=tag)

    def rename_torrent_file(self, _hash, old_path, new_path) -> bool:
        if False:
            return 10
        logger.info(f'{old_path} >> {new_path}')
        return self.client.torrents_rename_file(torrent_hash=_hash, old_path=old_path, new_path=new_path)

    def delete_torrent(self, hashes):
        if False:
            for i in range(10):
                print('nop')
        self.client.torrents_delete(hashes)
        logger.info('[Downloader] Remove torrents.')

    def add_torrent(self, torrent: Torrent | list, bangumi: Bangumi) -> bool:
        if False:
            while True:
                i = 10
        if not bangumi.save_path:
            bangumi.save_path = self._gen_save_path(bangumi)
        with RequestContent() as req:
            if isinstance(torrent, list):
                if len(torrent) == 0:
                    logger.debug(f'[Downloader] No torrent found: {bangumi.official_title}')
                    return False
                if 'magnet' in torrent[0].url:
                    torrent_url = [t.url for t in torrent]
                    torrent_file = None
                else:
                    torrent_file = [req.get_content(t.url) for t in torrent]
                    torrent_url = None
            elif 'magnet' in torrent.url:
                torrent_url = torrent.url
                torrent_file = None
            else:
                torrent_file = req.get_content(torrent.url)
                torrent_url = None
        if self.client.add_torrents(torrent_urls=torrent_url, torrent_files=torrent_file, save_path=bangumi.save_path, category='Bangumi'):
            logger.debug(f'[Downloader] Add torrent: {bangumi.official_title}')
            return True
        else:
            logger.debug(f'[Downloader] Torrent added before: {bangumi.official_title}')
            return False

    def move_torrent(self, hashes, location):
        if False:
            return 10
        self.client.move_torrent(hashes=hashes, new_location=location)

    def add_rss_feed(self, rss_link, item_path='Mikan_RSS'):
        if False:
            return 10
        self.client.rss_add_feed(url=rss_link, item_path=item_path)

    def remove_rss_feed(self, item_path):
        if False:
            print('Hello World!')
        self.client.rss_remove_item(item_path=item_path)

    def get_rss_feed(self):
        if False:
            print('Hello World!')
        return self.client.rss_get_feeds()

    def get_download_rules(self):
        if False:
            print('Hello World!')
        return self.client.get_download_rule()

    def get_torrent_path(self, hashes):
        if False:
            for i in range(10):
                print('nop')
        return self.client.get_torrent_path(hashes)

    def set_category(self, hashes, category):
        if False:
            print('Hello World!')
        self.client.set_category(hashes, category)

    def remove_rule(self, rule_name):
        if False:
            return 10
        self.client.remove_rule(rule_name)
        logger.info(f'[Downloader] Delete rule: {rule_name}')