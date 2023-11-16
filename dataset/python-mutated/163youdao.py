import time
from selenium import webdriver
login_url = 'http://account.youdao.com/login?service=dict'
xpaths = {'usernameTxtBox': ".//*[@id='username']", 'passwordTxtBox': ".//*[@id='password']", 'submitButton': ".//*[@id='login']/div[2]/div/div[1]/form/p[4]/nobr/input"}

def login():
    if False:
        return 10
    mydriver = webdriver.Firefox()
    mydriver.get(login_url)
    mydriver.maximize_window()
    mydriver.find_element_by_xpath(xpaths['usernameTxtBox']).clear()
    username = input('Please type your user name:\n')
    mydriver.find_element_by_xpath(xpaths['usernameTxtBox']).send_keys(username)
    mydriver.find_element_by_xpath(xpaths['passwordTxtBox']).clear()
    password = input('Please type your password:\n')
    mydriver.find_element_by_xpath(xpaths['passwordTxtBox']).send_keys(password)
    mydriver.find_element_by_xpath(xpaths['submitButton']).click()
    print('登录成功')
    time.sleep(5)
if __name__ == '__main__':
    login()