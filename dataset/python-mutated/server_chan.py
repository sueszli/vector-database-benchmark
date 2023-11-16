import logging
from module.models import Notification
from module.network import RequestContent
logger = logging.getLogger(__name__)

class ServerChanNotification(RequestContent):
    """Server酱推送"""

    def __init__(self, token, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        self.notification_url = f'https://sctapi.ftqq.com/{token}.send'

    @staticmethod
    def gen_message(notify: Notification) -> str:
        if False:
            for i in range(10):
                print('nop')
        text = f'\n        番剧名称：{notify.official_title}\n季度： 第{notify.season}季\n更新集数： 第{notify.episode}集\n{notify.poster_path}\n\n        '
        return text.strip()

    def post_msg(self, notify: Notification) -> bool:
        if False:
            return 10
        text = self.gen_message(notify)
        data = {'title': notify.official_title, 'desp': text}
        resp = self.post_data(self.notification_url, data)
        logger.debug(f'ServerChan notification: {resp.status_code}')
        return resp.status_code == 200