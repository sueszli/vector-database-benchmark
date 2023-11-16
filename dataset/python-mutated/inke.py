import requests

class InKe:

    def __init__(self, rid):
        if False:
            while True:
                i = 10
        self.rid = rid

    def get_real_url(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            room_url = 'https://webapi.busi.inke.cn/web/live_share_pc?uid=' + str(self.rid)
            response = requests.get(url=room_url).json()
            record_url = response.get('data').get('file').get('record_url')
            stream_addr = response.get('data').get('live_addr')
            real_url = {'record_url': record_url, 'stream_addr': stream_addr}
        except:
            raise Exception('直播间不存在或未开播')
        return real_url

def get_real_url(rid):
    if False:
        while True:
            i = 10
    try:
        inke = InKe(rid)
        return inke.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('请输入映客直播房间号：\n')
    print(get_real_url(r))