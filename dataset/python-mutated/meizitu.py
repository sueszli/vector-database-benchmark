import requests
import os
import time
import threading
from bs4 import BeautifulSoup

def download_page(url):
    if False:
        i = 10
        return i + 15
    '\n    用于下载页面\n    '
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0'}
    r = requests.get(url, headers=headers)
    r.encoding = 'gb2312'
    return r.text

def get_pic_list(html):
    if False:
        for i in range(10):
            print('nop')
    '\n    获取每个页面的套图列表,之后循环调用get_pic函数获取图片\n    '
    soup = BeautifulSoup(html, 'html.parser')
    pic_list = soup.find_all('li', class_='wp-item')
    for i in pic_list:
        a_tag = i.find('h3', class_='tit').find('a')
        link = a_tag.get('href')
        text = a_tag.get_text()
        get_pic(link, text)

def get_pic(link, text):
    if False:
        for i in range(10):
            print('nop')
    '\n    获取当前页面的图片,并保存\n    '
    html = download_page(link)
    soup = BeautifulSoup(html, 'html.parser')
    pic_list = soup.find('div', id='picture').find_all('img')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0'}
    create_dir('pic/{}'.format(text))
    for i in pic_list:
        pic_link = i.get('src')
        r = requests.get(pic_link, headers=headers)
        with open('pic/{}/{}'.format(text, pic_link.split('/')[-1]), 'wb') as f:
            f.write(r.content)
            time.sleep(1)

def create_dir(name):
    if False:
        print('Hello World!')
    if not os.path.exists(name):
        os.makedirs(name)

def execute(url):
    if False:
        print('Hello World!')
    page_html = download_page(url)
    get_pic_list(page_html)

def main():
    if False:
        for i in range(10):
            print('nop')
    create_dir('pic')
    queue = [i for i in range(1, 72)]
    threads = []
    while len(queue) > 0:
        for thread in threads:
            if not thread.is_alive():
                threads.remove(thread)
        while len(threads) < 5 and len(queue) > 0:
            cur_page = queue.pop(0)
            url = 'http://meizitu.com/a/more_{}.html'.format(cur_page)
            thread = threading.Thread(target=execute, args=(url,))
            thread.setDaemon(True)
            thread.start()
            print('{}正在下载{}页'.format(threading.current_thread().name, cur_page))
            threads.append(thread)
if __name__ == '__main__':
    main()