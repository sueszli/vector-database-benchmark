from urllib.parse import urlencode
import requests, pymysql
from pyquery import PyQuery as pq
from selenium import webdriver
from time import sleep
connection = pymysql.connect(host='localhost', port=3306, user='root', passwd='zkyr1006', db='python', charset='utf8')
cursor = connection.cursor()
sql = 'USE python;'
cursor.execute(sql)
connection.commit()
base_url = 'https://m.weibo.cn/api/container/getIndex?'
headers = {'Host': 'm.weibo.cn', 'Referer': 'https://m.weibo.cn/u/2145291155', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36', 'X-Requested-With': 'XMLHttpRequest'}

def create_sheet(bozhu):
    if False:
        print('Hello World!')
    try:
        weibo = '\n            CREATE TABLE weibo(\n                ID  VARCHAR (255) NOT NULL PRIMARY KEY,\n                text VARCHAR (255),\n                attitudes VARCHAR (255),\n                comments VARCHAR (255), \n                reposts VARCHAR (255) \n            )\n        '
        cursor.execute(weibo)
        connection.commit()
    except:
        pass

def url_get():
    if False:
        while True:
            i = 10
    browser = webdriver.PhantomJS()
    browser.get(url='https://m.weibo.cn/')
    wb_name = browser.find_element_by_class_name('W_input')
    wb_name.send_keys(input('输入博主ID：'))
    sleep(10)
    search = browser.find_element_by_class_name('W_ficon ficon_search S_ficon')
    search.click()
    sleep(5)
    bz_num = browser.find_element_by_class_name('name_txt')
    bz_num.click()
    sleep(5)
    handles = browser.window_handles
    browser.switch_to_window(handles[1])

def get_page(page):
    if False:
        return 10
    params = {'type': 'uid', 'value': '2145291155', 'containerid': '1076032145291155', 'page': page}
    url = base_url + urlencode(params)
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
    except requests.ConnectionError as e:
        print('Error', e.args)

def parse_page(json):
    if False:
        i = 10
        return i + 15
    if json:
        items = json.get('data').get('cards')
        for (index, item) in enumerate(items):
            if page == 1 and index == 1:
                continue
            else:
                item = item.get('mblog')
                weibo = []
                weibo.append(item.get('id'))
                weibo.append(pq(item.get('text')).text())
                weibo.append(item.get('attitudes_count'))
                weibo.append(item.get('comments_count'))
                weibo.append(item.get('reposts_count'))
                try:
                    sql = 'INSERT INTO weibo (ID,text,attitudes,comments,reposts)\n                          VALUES (%s,%s,%s,%s,%s) '
                    cursor.execute(sql, weibo)
                    connection.commit()
                except:
                    pass
            yield weibo
if __name__ == '__main__':
    for page in range(1, 17):
        json = get_page(page)
        results = parse_page(json)
        for result in results:
            print(result)
cursor.close()