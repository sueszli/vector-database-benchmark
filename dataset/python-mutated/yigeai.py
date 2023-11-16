"""
『一个AI』自动回复 (http://www.yige.ai/)
"""
import requests
from everyday_wechat.utils.common import is_json, md5_encode
from everyday_wechat.utils import config
__all__ = ['get_auto_reply', 'BOT_INDEX', 'BOT_NAME']
BOT_INDEX = 2
BOT_NAME = '一个 AI 机器人'
TULING_ERROR_CODE_LIST = ('501', '502', '503', '504', '507', '510')

def get_yigeai(text, userid):
    if False:
        print('Hello World!')
    '\n    『一个AI』自动回复 (http://www.yige.ai/)\n    接口说明：http://docs.yige.ai/Query%E6%8E%A5%E5%8F%A3.html\n    :param text:str, 需要发送的话\n    :userid:str,机器唯一标识\n    :return:str\n    '
    try:
        info = config.get('auto_reply_info')['yigeai_conf']
        token = info['client_token']
        if not token:
            print('一个「AI」token 为空,请求出错')
            return None
        session_id = md5_encode(userid if userid else '250')
        data = {'token': token, 'query': text, 'session_id': session_id}
        resp = requests.post('http://www.yige.ai/v1/query', data=data)
        if resp.status_code == 200 and is_json(resp):
            re_data = resp.json()
            code = re_data['status']['code']
            if code and str(code) not in TULING_ERROR_CODE_LIST:
                return_text = re_data['answer']
                return return_text
            error_text = re_data['status']['error_type']
            print('『一个AI』机器人错误信息：{}'.format(error_text))
            return None
        print('『一个AI』机器人获取数据失败')
    except Exception as e:
        print(e)
        print('『一个AI』机器人获取数据失败')
get_auto_reply = get_yigeai
if __name__ == '__main__':
    pass