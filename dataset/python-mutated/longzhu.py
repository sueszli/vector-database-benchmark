import requests
import re

class LongZhu:

    def __init__(self, rid):
        if False:
            for i in range(10):
                print('nop')
        '\n        龙珠直播，获取hls格式的播放地址\n        Args:\n            rid: 直播房间号\n        '
        self.rid = rid
        self.s = requests.Session()

    def get_real_url(self):
        if False:
            print('Hello World!')
        try:
            res = self.s.get(f'http://star.longzhu.com/{self.rid}').text
            roomId = re.search('roomid":(\\d+)', res).group(1)
            res = self.s.get(f'http://livestream.longzhu.com/live/getlivePlayurl?roomId={roomId}&utmSr=&platform=h5&device=ios').json()
            real_url = res.get('playLines')[0].get('urls')[-1].get('securityUrl')
        except Exception:
            raise Exception('直播间不存在或未开播')
        return real_url

def get_real_url(rid):
    if False:
        return 10
    try:
        lz = LongZhu(rid)
        return lz.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('请输入龙珠直播房间号：\n')
    print(get_real_url(r))