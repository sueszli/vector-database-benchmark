"""
https://github.com/MZCretin/RollToolsApi#随机获取笑话段子列表
随机获取笑话段子列表
"""
import requests
__all__ = ['get_rtjokes_info']

def get_rtjokes_info():
    if False:
        i = 10
        return i + 15
    '\n    随机获取笑话段子列表(https://github.com/MZCretin/RollToolsApi#随机获取笑话段子列表)\n    :return: str,笑话。\n    '
    print('获取随机笑话...')
    try:
        resp = requests.get('https://www.mxnzp.com/api/jokes/list/random')
        if resp.status_code == 200:
            content_dict = resp.json()
            if content_dict['code'] == 1:
                return_text = content_dict['data'][0]['content']
                return return_text
            else:
                print(content_dict['msg'])
        print('获取笑话失败。')
    except Exception as exception:
        print(exception)
        return None
    return None
get_one_words = get_rtjokes_info
if __name__ == '__main__':
    get_rtjokes_info()