from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import os
from pyquery import PyQuery as pq
from config import settings as SET
import re
browser_for_login = webdriver.Chrome()
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
browser = webdriver.Chrome(chrome_options=chrome_options)
wait = WebDriverWait(browser, 10)
total_num_of_products = SET['total_products']
total_num_of_products_cur = 0
choice_list = []
ban_list = []

def do_try(url):
    if False:
        print('Hello World!')
    try:
        browser.switch_to.window(browser.window_handles[1])
        browser.get(url)
        time.sleep(2)
        button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#product-intro > div.info > div.try-info.clearfix.bigImg > div.info-detail.chosen > div > div.btn-wrap > a')))
        if button.text != '申请试用':
            browser.switch_to.window(browser.window_handles[0])
            return False
        button.click()
        button2 = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'body > div.ui-dialog > div.ui-dialog-content > div > div > div.btn > a.y')))
        time.sleep(1)
        button2.click()
        time.sleep(2)
        browser.switch_to.window(browser.window_handles[0])
        return True
    except TimeoutException:
        browser.switch_to.window(browser.window_handles[0])
        return True

def get_try(page):
    if False:
        for i in range(10):
            print('nop')
    url = 'https://try.jd.com/activity/getActivityList' + '?page=' + str(page)
    browser.get(url)
    time.sleep(2)
    html = browser.page_source
    doc = pq(html)
    doc('.applied').remove()
    items = doc('.root61 .container .w .goods-list .items .con .clearfix .item').items()
    items = list(items)
    for item in items:
        title = item('.p-name').text()
        if check_name(title) == False:
            continue
        price_text = item('.p-price').text()[6:]
        price = float(price_text)
        if price < float(SET['price_limit']):
            continue
        try_url = 'https:' + item('.link').attr('href')
        print('价格: ', price)
        print(title)
        time.sleep(1)
        global total_num_of_products_cur
        global total_num_of_products
        if do_try(try_url) == True:
            total_num_of_products_cur += 1
            print('申请成功')
            print('')
        else:
            print('申请失败')
            print('')
        if total_num_of_products_cur >= total_num_of_products:
            return

def Control_try(total_page):
    if False:
        i = 10
        return i + 15
    browser.execute_script('window.open()')
    browser.switch_to.window(browser.window_handles[0])
    for page in range(1, total_page + 1):
        print('开始申请第' + str(page) + '页')
        get_try(page)
        global total_num_of_products
        global total_num_of_products_cur
        if total_num_of_products_cur >= total_num_of_products:
            return
        print('第' + str(page) + '页申请完成')

def login():
    if False:
        i = 10
        return i + 15
    browser_for_login.get('https://passport.jd.com/new/login.aspx')
    while browser_for_login.current_url != 'https://www.jd.com/':
        time.sleep(2)
    cookies = browser_for_login.get_cookies()
    browser_for_login.close()
    browser.get('https://www.jd.com')
    for cookie in cookies:
        browser.add_cookie(cookie)
    browser.get('https://www.jd.com')

def auto_showdown():
    if False:
        print('Hello World!')
    if SET['auto_shutdown'] == True:
        print('\n5秒后将自动关机')
        time.sleep(5)
        os.system('shutdown -s -t 1')

def deal_file():
    if False:
        for i in range(10):
            print('nop')
    global choice_list
    global ban_list
    if SET['choice'] == True:
        with open('choice.txt', 'r') as f:
            choice_list = re.split('[ |.|,|!|\n]', f.read())
            f.close()
    if SET['ban'] == True:
        with open('ban.txt', 'r') as f:
            ban_list = re.split('[ |.|,|!|\n]', f.read())
            f.close()

def check_name(title):
    if False:
        while True:
            i = 10
    is_choice = False
    if len(choice_list) == 0:
        is_choice = True
    for ch in choice_list:
        if ch in title:
            is_choice = True
            break
    if is_choice == False:
        return False
    is_ban = False
    for ba in ban_list:
        if ba in title:
            is_ban = True
            break
    if is_ban == True:
        return False
    return True
if __name__ == '__main__':
    deal_file()
    login()
    Control_try(SET['total_num_of_page'])
    browser.close()
    print('申请完成')
    auto_showdown()