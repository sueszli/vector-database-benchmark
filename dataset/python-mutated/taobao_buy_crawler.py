from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pyquery import PyQuery as pq
from time import sleep
import random

class taobao_infos:

    def __init__(self):
        if False:
            return 10
        url = 'https://login.taobao.com/member/login.jhtml'
        self.url = url
        options = webdriver.ChromeOptions()
        options.add_experimental_option('prefs', {'profile.managed_default_content_settings.images': 2})
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.browser = webdriver.Chrome(executable_path=chromedriver_path, options=options)
        self.wait = WebDriverWait(self.browser, 10)

    def login(self):
        if False:
            for i in range(10):
                print('nop')
        self.browser.get(self.url)
        self.browser.implicitly_wait(30)
        self.browser.find_element_by_xpath('//*[@class="forget-pwd J_Quick2Static"]').click()
        self.browser.implicitly_wait(30)
        self.browser.find_element_by_xpath('//*[@class="weibo-login"]').click()
        self.browser.implicitly_wait(30)
        self.browser.find_element_by_name('username').send_keys(weibo_username)
        self.browser.implicitly_wait(30)
        self.browser.find_element_by_name('password').send_keys(weibo_password)
        self.browser.implicitly_wait(30)
        self.browser.find_element_by_xpath('//*[@class="btn_tip"]/a/span').click()
        taobao_name = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.site-nav-bd > ul.site-nav-bd-l > li#J_SiteNavLogin > div.site-nav-menu-hd > div.site-nav-user > a.site-nav-login-info-nick ')))
        print(taobao_name.text)

    def swipe_down(self, second):
        if False:
            while True:
                i = 10
        for i in range(int(second / 0.1)):
            if i % 2 == 0:
                js = 'var q=document.documentElement.scrollTop=' + str(300 + 400 * i)
            else:
                js = 'var q=document.documentElement.scrollTop=' + str(200 * i)
            self.browser.execute_script(js)
            sleep(0.1)
        js = 'var q=document.documentElement.scrollTop=100000'
        self.browser.execute_script(js)
        sleep(0.1)

    def crawl_good_buy_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.browser.get('https://buyertrade.taobao.com/trade/itemlist/list_bought_items.htm')
        for page in range(1, 1000):
            good_total = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#tp-bought-root > div.js-order-container')))
            html = self.browser.page_source
            doc = pq(html)
            good_items = doc('#tp-bought-root .js-order-container').items()
            for item in good_items:
                good_time_and_id = item.find('.bought-wrapper-mod__head-info-cell___29cDO').text().replace('\n', '').replace('\r', '')
                good_merchant = item.find('.seller-mod__container___1w0Cx').text().replace('\n', '').replace('\r', '')
                good_name = item.find('.sol-mod__no-br___1PwLO').text().replace('\n', '').replace('\r', '')
                print(good_time_and_id, good_merchant, good_name)
            print('\n\n')
            swipe_time = random.randint(1, 3)
            self.swipe_down(swipe_time)
            good_total = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.pagination-next')))
            good_total.click()
            sleep(2)
if __name__ == '__main__':
    chromedriver_path = '/Users/bird/Desktop/chromedriver.exe'
    weibo_username = '改成你的微博账号'
    weibo_password = '改成你的微博密码'
    a = taobao_infos()
    a.login()
    a.crawl_good_buy_data()