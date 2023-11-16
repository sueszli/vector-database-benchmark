import requests

class MHT:

    def __init__(self, rid):
        if False:
            print('Hello World!')
        self.rid = rid

    def get_real_url(self):
        if False:
            return 10
        with requests.Session() as s:
            res = s.get(f'https://www.2cq.com/proxy/room/room/info?roomId={self.rid}&appId=1004')
        res = res.json()
        if res['status'] == 1:
            result = res['result']
            if result['liveState'] == 1:
                real_url = result['pullUrl']
                return real_url
            else:
                raise Exception('未开播')
        else:
            raise Exception('直播间可能不存在')

def get_real_url(rid):
    if False:
        return 10
    try:
        mht = MHT(rid)
        return mht.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('输入棉花糖直播房间号：\n')
    print(get_real_url(r))