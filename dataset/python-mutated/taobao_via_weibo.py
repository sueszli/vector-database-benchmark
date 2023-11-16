"""
author : CriseLYJ
github : https://github.com/CriseLYJ/
update_time : 2019-4-2
"""
'\n淘宝登陆有时候不会跳出二维码页面，如果失败，请重新运行程序即可\n'
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class Taobao_Spider:

    def __init__(self, username, password):
        if False:
            for i in range(10):
                print('nop')
        '初始化参数'
        url = 'https://login.taobao.com/member/login.jhtml'
        self.url = url
        options = webdriver.ChromeOptions()
        options.add_experimental_option('prefs', {'profile.managed_default_content_settings.images': 2})
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.browser = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.browser, 10)
        self.username = username
        self.password = password

    def run(self):
        if False:
            i = 10
            return i + 15
        '登陆接口'
        self.browser.get(self.url)
        try:
            login_element = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.qrcode-login > .login-links > .forget-pwd')))
            login_element.click()
            sina_login = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.weibo-login')))
            sina_login.click()
            weibo_user = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.username > .W_input')))
            weibo_user.send_keys(self.username)
            sina_password = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.password > .W_input')))
            sina_password.send_keys(self.password)
            submit = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.btn_tip > a > span')))
            submit.click()
            taobao_name = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.site-nav-bd > ul.site-nav-bd-l > li#J_SiteNavLogin > div.site-nav-menu-hd > div.site-nav-user > a.site-nav-login-info-nick ')))
            print('登陆成功：%s' % taobao_name.text)
        except Exception:
            self.browser.close()
            print('登陆失败')
if __name__ == '__main__':
    username = input('请输入你的微博用户名:')
    password = input('请输入密码:')
    spider = Taobao_Spider(username, password)
    spider.run()