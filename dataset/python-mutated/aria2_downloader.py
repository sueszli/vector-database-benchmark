import logging
import time
from aria2p import API, Client, ClientException
from module.conf import settings
logger = logging.getLogger(__name__)

class QbDownloader:

    def __init__(self, host, username, password):
        if False:
            i = 10
            return i + 15
        while True:
            try:
                self._client = API(Client(host=host, port=6800, secret=password))
                break
            except ClientException:
                logger.warning(f"Can't login Aria2 Server {host} by {username}, retry in {settings.connect_retry_interval}")
            time.sleep(settings.connect_retry_interval)

    def torrents_add(self, urls, save_path, category):
        if False:
            i = 10
            return i + 15
        return self._client.add_torrent(is_paused=settings.dev_debug, torrent_file_path=urls, save_path=save_path, category=category)