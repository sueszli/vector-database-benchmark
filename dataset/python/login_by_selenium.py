# tested on ubuntu15.04
import time
from selenium import webdriver

login_url = 'https://passport.jd.com/new/login.aspx'
driver = webdriver.PhantomJS()
driver.get(login_url)
time.sleep(5)

account = driver.find_element_by_id('loginname')
password = driver.find_element_by_id('nloginpwd')
submit = driver.find_element_by_id('loginsubmit')

account.clear()
password.clear()
account.send_keys('yourname')
password.send_keys('yourpassword')

submit.click()
time.sleep(5)

# cookie和前面一样的方式获取和保存
cookies = driver.get_cookies()
driver.close()
