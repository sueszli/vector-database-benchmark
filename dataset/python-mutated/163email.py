import time
from getpass import getpass
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

def login():
    if False:
        i = 10
        return i + 15
    acount_num = input('请输入账号:')
    passwd_str = getpass('请输入密码:')
    driver = webdriver.Chrome()
    url = 'http://mail.163.com/'
    driver.get(url)
    wait = WebDriverWait(driver, 10)
    wait.until(EC.element_to_be_clickable((By.ID, 'lbNormal')))
    driver.find_element_by_id('lbNormal').click()
    elem = driver.find_element_by_css_selector("iframe[id^='x-URS-iframe']")
    driver.switch_to.frame(elem)
    account_el = driver.find_element_by_xpath('//input[@name="email"]')
    account_el.clear()
    account_el.send_keys(acount_num)
    password_el = driver.find_element_by_xpath('//input[@name="password"]')
    password_el.clear()
    password_el.send_keys(passwd_str)
    login_el = driver.find_element_by_xpath('//a[@id="dologin"]')
    login_el.click()
    time.sleep(10)
    cur_cookies = driver.get_cookies()
    return cur_cookies
if __name__ == '__main__':
    login()