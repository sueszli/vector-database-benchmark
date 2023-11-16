import requests
'\ninfo:\nauthor:CriseLYJ\ngithub:https://github.com/CriseLYJ/\nupdate_time:2019-04-06\n'
'\n模拟登陆招聘狗\n'

class ZhaoPinGouLogin(object):

    def __init__(self, account, password):
        if False:
            return 10
        self.url = 'https://qiye.zhaopingou.com/zhaopingou_interface/security_login?timestamp=1554552162122'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36', 'Refer': 'https://qiye.zhaopingou.com/signin?callback=https%3A%2F%2Fqiye.zhaopingou.com%2Fresume', 'Host': 'qiye.zhaopingou.com'}
        self.data = {'userName': account, 'password': password, 'code': '', 'clientNo': '', 'userToken': '', 'clientType': '2'}
        self.session = requests.Session()

    def get_coolie(self):
        if False:
            while True:
                i = 10
        '模拟登陆获取cookie'
        resp = self.session.post(url=self.url, headers=self.headers, data=self.data)
        resp_dict = resp.json()
        if resp_dict['errorCode'] == 1:
            print('登陆成功')
            cookies = resp.cookies
            print(cookies)
            return cookies
        else:
            print('登陆失败')

    def run(self):
        if False:
            print('Hello World!')
        self.get_coolie()
if __name__ == '__main__':
    account = input('请输入你的账号：')
    password = input('请输入你的密码：')
    spider = ZhaoPinGouLogin(account, password)
    spider.run()