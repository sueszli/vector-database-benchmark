import json
import re
import requests

class KuaiShou:

    def __init__(self, rid):
        if False:
            while True:
                i = 10
        self.rid = rid

    def get_real_url(self):
        if False:
            i = 10
            return i + 15
        headers = {'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1', 'cookie': 'did=web_d563dca728d28b00336877723e0359ed'}
        with requests.Session() as s:
            res = s.get('https://m.gifshow.com/fw/live/{}'.format(self.rid), headers=headers)
            livestream = re.search('liveStream":(.*),"obfuseData', res.text)
            if livestream:
                livestream = json.loads(livestream.group(1))
                (*_, hlsplayurls) = livestream['multiResolutionHlsPlayUrls']
                (urls,) = hlsplayurls['urls']
                url = urls['url']
                return url
            else:
                raise Exception('直播间不存在或未开播')

def get_real_url(rid):
    if False:
        for i in range(10):
            print('nop')
    try:
        ks = KuaiShou(rid)
        return ks.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('请输入快手直播房间ID：\n')
    print(get_real_url(r))