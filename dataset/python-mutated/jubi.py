__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
import random
import hashlib
import hmac, time
import smtplib
from email.mime.text import MIMEText
from email import Utils
import threading
import requests, itchat
from toolkit import Toolkit

class Jubi_web:

    def __init__(self, send=None):
        if False:
            print('Hello World!')
        cfg = Toolkit.getUserData('data.cfg')
        self.public_key = cfg['public_key']
        self.private_key = cfg['private_key']
        self.send = send
        from_mail = cfg['from_mail']
        password = cfg['password']
        to_mail = cfg['to_mail']
        smtp_server = 'smtp.qq.com'
        self.server = smtp_server
        self.username = from_mail.split('@')[0]
        self.from_mail = from_mail
        self.password = password
        self.to_mail = to_mail
        self.coin_list = ['IFC', 'DOGE', 'EAC', 'DNC', 'MET', 'ZET', 'SKT', 'YTC', 'PLC', 'LKC', 'JBC', 'MRYC', 'GOOC', 'QEC', 'PEB', 'XRP', 'NXT', 'WDC', 'MAX', 'ZCC', 'HLB', 'RSS', 'PGC', 'RIO', 'XAS', 'TFC', 'BLK', 'FZ', 'ANS', 'XPM', 'VTC', 'KTC', 'VRC', 'XSGS', 'LSK', 'PPC', 'ETC', 'GAME', 'LTC', 'ETH', 'BTC']
        if self.send == 'msn':
            try:
                self.smtp = smtplib.SMTP_SSL(port=465)
                self.smtp.connect(self.server)
                self.smtp.login(self.username, self.password)
            except smtplib.SMTPException as e:
                print(e)
                return 0
        if send == 'wechat':
            self.w_name = 'wwwei'
            self.w_name1 = 'aiweichuangyi'
            itchat.auto_login(hotReload=True)
            account = itchat.get_friends(self.w_name)
            for i in account:
                if i['PYQuanPin'] == self.w_name:
                    self.toName = i['UserName']
                if i['PYQuanPin'] == self.w_name1:
                    self.toName1 = i['UserName']

    def send_wechat(self, name, content, user):
        if False:
            for i in range(10):
                print('nop')
        w_content = name + ' ' + content
        itchat.send(w_content, toUserName=user)
        time.sleep(1)
        itchat.send(w_content, toUserName='filehelper')

    def send_text(self, name, content):
        if False:
            for i in range(10):
                print('nop')
        subject = '%s' % name
        self.msg = MIMEText(content, 'plain', 'utf-8')
        self.msg['to'] = self.to_mail
        self.msg['from'] = self.from_mail
        self.msg['Subject'] = subject
        self.msg['Date'] = Utils.formatdate(localtime=1)
        try:
            self.smtp.sendmail(self.msg['from'], self.msg['to'], self.msg.as_string())
            self.smtp.quit()
            print('sent')
        except smtplib.SMTPException as e:
            print(e)
            return 0

    def warming(self, coin, up_price, down_price, user):
        if False:
            i = 10
            return i + 15
        url = 'https://www.jubi.com/api/v1/ticker/'
        while 1:
            time.sleep(5)
            try:
                data = requests.post(url, data={'coin': coin}).json()
            except Exception as e:
                print(e)
                print('time out. Retry')
                time.sleep(15)
                continue
            current = float(data['last'])
            if current >= up_price:
                print('Up to ', up_price)
                print('current price ', current)
                if self.send == 'msn':
                    self.send_text(coin, str(current))
                if self.send == 'wechat':
                    self.send_wechat(coin, str(current), user)
                time.sleep(1200)
            if current <= down_price:
                print('Down to ', down_price)
                print('current price ', current)
                if self.send == 'msn':
                    self.send_text(coin, str(current))
                if self.send == 'wechat':
                    self.send_wechat(coin, str(current), user)
                time.sleep(1200)

    def getContent(self):
        if False:
            i = 10
            return i + 15
        url = 'https://www.jubi.com/api/v1/trade_list'
        params_data = {'key': 'x', 'signature': 'x'}
        s = requests.get(url=url, params=params_data)

    def getHash(self, s):
        if False:
            print('Hello World!')
        m = hashlib.md5()
        m.update(s)
        return m.hexdigest()

    def sha_convert(self, s):
        if False:
            while True:
                i = 10
        return hashlib.sha256(self.getHash(s)).hexdigest()

    def get_nonce(self):
        if False:
            i = 10
            return i + 15
        lens = 12
        return ''.join([str(random.randint(0, 9)) for i in range(lens)])

    def get_signiture(self):
        if False:
            for i in range(10):
                print('nop')
        url = 'https://www.jubi.com/api/v1/ticker/'
        coin = 'zet'
        nonce = self.get_nonce()
        md5 = self.getHash(self.private_key)
        message = 'nonce=' + nonce + '&' + 'key=' + self.public_key
        signature = hmac.new(md5, message, digestmod=hashlib.sha256).digest()
        req = requests.post(url, data={'coin': coin})
        print(req.status_code)
        print(req.text)

    def real_time_ticker(self, coin):
        if False:
            i = 10
            return i + 15
        url = 'https://www.jubi.com/api/v1/ticker/'
        try:
            data = requests.post(url, data={'coin': coin}).json()
        except Exception as e:
            print(e)
            data = None
        return data

    def real_time_depth(self, coin):
        if False:
            print('Hello World!')
        url = 'https://www.jubi.com/api/v1/depth/'
        data = requests.post(url, data={'coin': coin}).json()
        print(data)
        data_bids = data['bids']
        data_asks = data['asks']
        print('bids')
        for i in data_bids:
            print(i[0])
            print(' ')
            print(i[1])
        print('asks')
        for j in data_asks:
            print(j[0])
            print(' ')
            print(j[1])

    def list_all_price(self):
        if False:
            while True:
                i = 10
        for i in self.coin_list:
            print(i)
            print(' price: ')
            p = self.real_time_ticker(i.lower())
            if p is not None:
                print(p['last'])

    def getOrder(self, coin):
        if False:
            while True:
                i = 10
        url = 'https://www.jubi.com/api/v1/orders/'
        try:
            req = requests.get(url, params={'coin': coin})
        except Exception as e:
            print(e)
        data = req.json()
        return data

    def turnover(self, coin):
        if False:
            print('Hello World!')
        i = coin.lower()
        coins = Toolkit.getUserData('coins.csv')
        total = long(coins[i])
        p = self.getOrder(i)
        print(p)
        amount = 0.0
        for j in p:
            t = j['amount']
            amount = float(t) + amount
        turn_over = amount * 1.0 / total * 100
        print(turn_over)

    def multi_thread(self, coin_list, price_list, username):
        if False:
            i = 10
            return i + 15
        thread_num = len(coin_list)
        thread_list = []
        for i in range(thread_num):
            if username[i] == 0:
                nameID = self.toName
            if username[i] == 1:
                nameID = self.toName1
            t = threading.Thread(target=self.warming, args=(coin_list[i], price_list[i][0], price_list[i][1], nameID))
            thread_list.append(t)
        for j in thread_list:
            j.start()
        for k in thread_list:
            k.join()
if __name__ == '__main__':
    obj = Jubi_web(send='wechat')
    coin_list = ['zet', 'doge']
    price_list = [[0.2, 0.11], [0.03, 0.02]]
    username = [0, 0]
    obj.multi_thread(coin_list, price_list, username)