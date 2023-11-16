import requests

class Live17:

    def __init__(self, rid):
        if False:
            return 10
        '\n        # 可能需要挂代理。\n        # self.proxies = {\n        #     "http": "http://xxxx:1080",\n        #     "https": "http://xxxx:1080",\n        # }\n        Args:\n            rid:\n        '
        self.rid = rid
        self.BASE_URL = 'https://api-dsa.17app.co/api/v1/lives/'

    def get_real_url(self):
        if False:
            i = 10
            return i + 15
        try:
            res = requests.get(f'{self.BASE_URL}{self.rid}').json()
            real_url_default = res.get('rtmpUrls')[0].get('url')
            real_url_modify = real_url_default.replace('global-pull-rtmp.17app.co', 'china-pull-rtmp-17.tigafocus.com')
            real_url = [real_url_modify, real_url_default]
        except Exception:
            raise Exception('直播间不存在或未开播')
        return real_url

def get_real_url(rid):
    if False:
        print('Hello World!')
    try:
        live17 = Live17(rid)
        return live17.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('请输入17直播房间号：\n')
    print(get_real_url(r))