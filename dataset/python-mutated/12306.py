"""
@author: liuyw
"""
from splinter.browser import Browser
from time import sleep
import traceback
import time, sys

class huoche(object):
    driver_name = ''
    executable_path = ''
    username = u'xxx'
    passwd = u'xxx'
    starts = u'%u6C88%u9633%2CSYT'
    ends = u'%u54C8%u5C14%u6EE8%2CHBB'
    dtime = u'2018-01-19'
    order = 0
    users = [u'xxx', u'xxx']
    xb = u'二等座'
    pz = u'成人票'
    '网址'
    ticket_url = 'https://kyfw.12306.cn/otn/leftTicket/init'
    login_url = 'https://kyfw.12306.cn/otn/login/init'
    initmy_url = 'https://kyfw.12306.cn/otn/index/initMy12306'
    buy = 'https://kyfw.12306.cn/otn/confirmPassenger/initDc'

    def __init__(self):
        if False:
            return 10
        self.driver_name = 'chrome'
        self.executable_path = 'D:/chromedriver'

    def login(self):
        if False:
            i = 10
            return i + 15
        self.driver.visit(self.login_url)
        self.driver.fill('loginUserDTO.user_name', self.username)
        self.driver.fill('userDTO.password', self.passwd)
        print(u'等待验证码，自行输入...')
        while True:
            if self.driver.url != self.initmy_url:
                sleep(1)
            else:
                break

    def start(self):
        if False:
            i = 10
            return i + 15
        self.driver = Browser(driver_name=self.driver_name, executable_path=self.executable_path)
        self.driver.driver.set_window_size(1400, 1000)
        self.login()
        self.driver.visit(self.ticket_url)
        try:
            print(u'购票页面开始...')
            self.driver.cookies.add({'_jc_save_fromStation': self.starts})
            self.driver.cookies.add({'_jc_save_toStation': self.ends})
            self.driver.cookies.add({'_jc_save_fromDate': self.dtime})
            self.driver.reload()
            count = 0
            if self.order != 0:
                while self.driver.url == self.ticket_url:
                    self.driver.find_by_text(u'查询').click()
                    count += 1
                    print(u'循环点击查询... 第 %s 次' % count)
                    try:
                        self.driver.find_by_text(u'预订')[self.order - 1].click()
                    except Exception as e:
                        print(e)
                        print(u'还没开始预订')
                        continue
            else:
                while self.driver.url == self.ticket_url:
                    self.driver.find_by_text(u'查询').click()
                    count += 1
                    print(u'循环点击查询... 第 %s 次' % count)
                    try:
                        for i in self.driver.find_by_text(u'预订'):
                            i.click()
                            sleep(1)
                    except Exception as e:
                        print(e)
                        print(u'还没开始预订 %s' % count)
                        continue
            print(u'开始预订...')
            sleep(1)
            print(u'开始选择用户...')
            for user in self.users:
                self.driver.find_by_text(user).last.click()
            print(u'提交订单...')
            sleep(1)
            self.driver.find_by_text(self.pz).click()
            self.driver.find_by_id('').select(self.pz)
            self.driver.find_by_text(self.xb).click()
            sleep(1)
            self.driver.find_by_id('submitOrder_id').click()
            print(u'开始选座...')
            self.driver.find_by_id('1D').last.click()
            self.driver.find_by_id('1F').last.click()
            sleep(1.5)
            print(u'确认选座...')
            self.driver.find_by_id('qr_submit_id').click()
        except Exception as e:
            print(e)
if __name__ == '__main__':
    huoche = huoche()
    huoche.start()