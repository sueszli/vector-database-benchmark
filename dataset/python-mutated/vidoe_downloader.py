import requests, re, json, sys
from bs4 import BeautifulSoup
from urllib import request

class video_downloader:

    def __init__(self, url):
        if False:
            while True:
                i = 10
        self.server = 'http://api.xfsub.com'
        self.api = 'http://api.xfsub.com/xfsub_api/?url='
        self.get_url_api = 'http://api.xfsub.com/xfsub_api/url.php'
        self.url = url.split('#')[0]
        self.headers = {'Referer': 'http://api.xfsub.com/xfsub_api/?url=%s?qqdrsign=055a4' % self.url}
        self.target = self.api + self.url
        self.s = requests.session()
    '\n\t函数说明:获取key、time、url等参数\n\tParameters:\n\t\t无\n\tReturns:\n\t\t无\n\tModify:\n\t\t2017-09-18\n\t'

    def get_key(self):
        if False:
            for i in range(10):
                print('nop')
        req = self.s.get(url=self.target)
        req.encoding = 'utf-8'
        self.info = json.loads(re.findall('"url.php",\\ (.+),', req.text)[0])
    '\n\t函数说明:获取视频地址\n\tParameters:\n\t\t无\n\tReturns:\n\t\tvideo_url - 视频存放地址\n\tModify:\n\t\t2017-09-18\n\t'

    def get_url(self):
        if False:
            i = 10
            return i + 15
        data = {'time': self.info['time'], 'key': self.info['key'], 'url': self.info['url'], 'type': ''}
        req = self.s.post(url=self.get_url_api, data=data, headers=self.headers)
        url = self.server + json.loads(req.text)['url']
        req = self.s.get(url=url, headers=self.headers)
        bf = BeautifulSoup(req.text, 'xml')
        video_url = bf.find('file').string
        return video_url
    '\n\t函数说明:回调函数，打印下载进度\n\tParameters:\n\t\ta b c - 返回信息\n\tReturns:\n\t\t无\n\tModify:\n\t\t2017-09-18\n\t'

    def Schedule(self, a, b, c):
        if False:
            while True:
                i = 10
        per = 100.0 * a * b / c
        if per > 100:
            per = 1
        sys.stdout.write('  ' + '%.2f%% 已经下载的大小:%ld 文件大小:%ld' % (per, a * b, c) + '\r')
        sys.stdout.flush()
    '\n\t函数说明:视频下载\n\tParameters:\n\t\turl - 视频地址\n\t\tfilename - 视频名字\n\tReturns:\n\t\t无\n\tModify:\n\t\t2017-09-18\n\t'

    def video_download(self, url, filename):
        if False:
            for i in range(10):
                print('nop')
        request.urlretrieve(url=url, filename=filename, reporthook=self.Schedule)
if __name__ == '__main__':
    url = 'http://www.iqiyi.com/v_19rr7qhfg0.html#vfrm=19-9-0-1'
    vd = video_downloader(url)
    filename = '加勒比海盗5'
    print('%s下载中:' % filename)
    vd.get_key()
    video_url = vd.get_url()
    print('  获取地址成功:%s' % video_url)
    vd.video_download(video_url, filename + '.mp4')
    print('\n下载完成！')