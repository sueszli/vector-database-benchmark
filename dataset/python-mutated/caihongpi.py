"""
https://chp.shadiao.app/?from_nmsl
彩虹屁生成器
 """
import requests
__all__ = ['get_caihongpi_info']

def get_caihongpi_info():
    if False:
        for i in range(10):
            print('nop')
    '\n    彩虹屁生成器\n    :return: str,彩虹屁\n    '
    print('获取彩虹屁信息...')
    try:
        resp = requests.get('https://chp.shadiao.app/api.php')
        if resp.status_code == 200:
            return resp.text
        print('彩虹屁获取失败。')
    except requests.exceptions.RequestException as exception:
        print(exception)
get_one_words = get_caihongpi_info
if __name__ == '__main__':
    ow = get_one_words()
    print(ow)
    pass