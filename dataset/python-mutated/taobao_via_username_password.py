import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
from getpass import getpass
'\n运行前必须要做的事情：\n如果直接使用webdriver，不做任何修改的话，淘宝可以断定启动的浏览器是“机器人”，而不是“死的机器”。\n如果想让淘宝错误地认为启动的浏览器是"死的机器"，那么就需要修改webdriver。\n我使用的是chromedriver，"perl -pi -e \'s/cdc_/dog_/g\' /usr/local/bin/chromedriver"是修改chromedriver的代码，直接在Terminal执行即可。执行完在运行此脚本，则可以成功登录。\n\n这里我解释一下"perl -pi -e \'s/cdc_/dog_/g\' /usr/local/bin/chromedriver"，这段代码其实就是全局修改/usr/local/bin/chromedriver中的cdc_为dog_，"/usr/local/bin/chromedriver"是chromedriver所在的文件路径。\n感谢https://www.jianshu.com/p/368be2cc6ca1这篇文章的作者。\n\n######################################\n- 已经修改的 webdriver 在仓库中请自行下载\n- 不保证所有的版本都可用，以下是我用的版本，如果不适应，请下载对应的版本自行修改\n- 另外感谢提供思路--\nversion: 版本 76.0.3809.100（正式版本） （64 位）\n######################################\n'

class TaobaoSpider:

    def __init__(self, username, password):
        if False:
            return 10
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option('prefs', {'profile.managed_default_content_settings.images': 2})
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.web_driver = webdriver.Chrome(options=chrome_options)
        self.web_driver_wait = WebDriverWait(self.web_driver, 10)
        self.url = 'https://login.taobao.com/member/login.jhtml'
        self.username = username
        self.password = password

    def login(self):
        if False:
            for i in range(10):
                print('nop')
        self.web_driver.get(self.url)
        try:
            login_method_switch = self.web_driver_wait.until(expected_conditions.presence_of_element_located((By.XPATH, '//*[@id="J_QRCodeLogin"]/div[5]/a[1]')))
            login_method_switch.click()
            username_input = self.web_driver_wait.until(expected_conditions.presence_of_element_located((By.ID, 'TPL_username_1')))
            username_input.send_keys(self.username)
            password_input = self.web_driver_wait.until(expected_conditions.presence_of_element_located((By.ID, 'TPL_password_1')))
            password_input.send_keys(self.password)
            login_button = self.web_driver_wait.until(expected_conditions.presence_of_element_located((By.XPATH, '//*[@id="J_SubmitStatic"]')))
            login_button.click()
            taobao_name_tag = self.web_driver_wait.until(expected_conditions.presence_of_element_located((By.XPATH, '//*[@id="J_Col_Main"]/div/div[1]/div/div[1]/div[1]/div/div[1]/a/em')))
            print(f'登陆成功：{taobao_name_tag.text}')
            time.sleep(5)
            self.web_driver.close()
        except Exception as e:
            print(e)
            self.web_driver.close()
            print('登陆失败')
if __name__ == '__main__':
    username = input('请输入用户名：')
    password = getpass('请输入密码：')
    spider = TaobaoSpider(username, password)
    spider.login()