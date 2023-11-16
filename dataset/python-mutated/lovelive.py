"""
从土味情话中获取每日一句。
 """
import requests
__all__ = ['get_lovelive_info']

def get_lovelive_info():
    if False:
        for i in range(10):
            print('nop')
    '\n    从土味情话中获取每日一句。\n    :return: str,土味情话。\n    '
    print('获取土味情话...')
    try:
        resp = requests.get('https://api.lovelive.tools/api/SweetNothings')
        if resp.status_code == 200:
            return resp.text
        print('土味情话获取失败。')
    except requests.exceptions.RequestException as exception:
        print(exception)
    return None
get_one_words = get_lovelive_info