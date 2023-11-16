from urllib.parse import quote
import requests
from wechatpy.client.api.base import BaseWeChatAPI

class WeChatQRCode(BaseWeChatAPI):

    def create(self, qrcode_data):
        if False:
            i = 10
            return i + 15
        '\n        创建二维码\n\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Account_Management/Generating_a_Parametric_QR_Code.html\n\n        :param qrcode_data: 你要发送的参数 dict\n        :return: 返回的 JSON 数据包\n\n        使用示例::\n\n        >>>    from wechatpy import WeChatClient\n        >>>\n        >>>    client = WeChatClient(\'appid\', \'secret\')\n        >>>    res = client.qrcode.create({\n        >>>        \'expire_seconds\': 1800,\n        >>>        \'action_name\': \'QR_SCENE\',\n        >>>        \'action_info\': {\n        >>>            \'scene\': {\'scene_id\': 123},\n        >>>        }\n        >>>    })\n        >>>    # 创建永久的二维码, 参数使用字符串而不是数字id\n        >>>    res = client.qrcode.create({\n        >>>        \'action_name\': \'QR_LIMIT_STR_SCENE\',\n        >>>        \'action_info\': {\n        >>>            \'scene\': {\'scene_str\': "scan_qrcode_from_scene"},\n        >>>        }\n        >>>    })\n\n        '
        return self._post('qrcode/create', data=qrcode_data)

    def show(self, ticket):
        if False:
            while True:
                i = 10
        "\n        通过ticket换取二维码\n\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Account_Management/Generating_a_Parametric_QR_Code.html\n\n        :param ticket: 二维码 ticket 。可以通过 :func:`create` 获取到\n        :return: 返回的 Request 对象\n\n        使用示例::\n\n        >>>    from wechatpy import WeChatClient\n        >>>\n        >>>    client = WeChatClient('appid', 'secret')\n        >>>    res = client.qrcode.show('ticket data')\n\n        "
        if isinstance(ticket, dict):
            ticket = ticket['ticket']
        return requests.get(url='https://mp.weixin.qq.com/cgi-bin/showqrcode', params={'ticket': ticket})

    @classmethod
    def get_url(cls, ticket):
        if False:
            return 10
        "\n        通过ticket换取二维码地址\n\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Account_Management/Generating_a_Parametric_QR_Code.html\n\n        :param ticket: 二维码 ticket 。可以通过 :func:`create` 获取到\n        :return: 返回的二维码地址\n\n        使用示例::\n\n        >>>    from wechatpy import WeChatClient\n        >>>\n        >>>    client = WeChatClient('appid', 'secret')\n        >>>    url = client.qrcode.get_url('ticket data')\n\n        "
        if isinstance(ticket, dict):
            ticket = ticket['ticket']
        ticket = quote(ticket)
        return f'https://mp.weixin.qq.com/cgi-bin/showqrcode?ticket={ticket}'