import sys
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def get_chrome():
    if False:
        i = 10
        return i + 15
    driver = webdriver.Chrome()
    print('beging_0')
    url = 'https://www.reg007.com/account/signin'
    driver.get(url)
    elem = driver.find_element_by_xpath('//*[@id="signin_email"]').send_keys('zhanghao@qq.com')
    elem = driver.find_element_by_xpath('//*[@id="signin_password"]').send_keys('mima')
    elem = driver.find_element_by_xpath('//*[@id="signin_form"]/button').click()
    time.sleep(1)
    elem = driver.find_element_by_xpath('/html/body/div[2]/div/div[2]/ol/li[1]/a').click()
    elem = driver.find_element_by_xpath('//*[@id="e_m"]')
    elem.send_keys('18322867654')
    time.sleep(1)
    elem = driver.find_element_by_xpath('//*[@id="tsb"]').click()
    time.sleep(10)
    with open('code.html', 'wb') as f:
        f.write(driver.page_source.encode('utf8'))
    time.sleep(5)
    driver.quit()
if __name__ == '__main__':
    get_chrome()