import requests
from urllib.parse import urlencode
from requests import codes
import os
from multiprocessing.pool import Pool
from bs4 import BeautifulSoup as bsp
import json
import time
import re
'\ninfo:\nauthor:CriseLYJ\ngithub:https://github.com/CriseLYJ/\nupdate_time:2019-3-7\n\n'

def get_index(offset):
    if False:
        while True:
            i = 10
    base_url = 'http://www.guokr.com/apis/minisite/article.json?'
    data = {'retrieve_type': 'by_subject', 'limit': '20', 'offset': offset}
    url = base_url + urlencode(data)
    try:
        resp = requests.get(url)
        if codes.ok == resp.status_code:
            return resp.json()
    except requests.ConnectionError:
        return None

def get_url(json):
    if False:
        for i in range(10):
            print('nop')
    if json.get('result'):
        result = json.get('result')
        for item in result:
            if item.get('cell_type') is not None:
                continue
            yield item.get('url')
    "\n    try:\n        result=json.load(json)\n        if result:\n            for i in result.get('result'):\n                yield i.get('url')\n    "

def get_text(url):
    if False:
        i = 10
        return i + 15
    html = requests.get(url).text
    print(html)
    soup = bsp(html, 'lxml')
    title = soup.find('h1', id='articleTitle').get_text()
    autor = soup.find('div', class_='content-th-info').find('a').get_text()
    article_content = soup.find('div', class_='document').find_all('p')
    all_p = [i.get_text() for i in article_content if not i.find('img') and (not i.find('a'))]
    article = '\n'.join(all_p)
    yield {'title': title, 'autor': autor, 'article': article}

def save_article(content):
    if False:
        i = 10
        return i + 15
    try:
        if content.get('title'):
            file_name = str(content.get('title')) + '.txt'
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write('\n'.join([str(content.get('title')), str(content.get('autor')), str(content.get('article'))]))
                print('Downloaded article path is %s' % file_name)
        else:
            file_name = str(content.get('title')) + '.txt'
            print('Already Downloaded', file_name)
    except requests.ConnectionError:
        print('Failed to Save Image，item %s' % content)

def main(offset):
    if False:
        print('Hello World!')
    result = get_index(offset)
    all_url = get_url(result)
    for url in all_url:
        article = get_text(url)
        for art in article:
            save_article(art)
GROUP_START = 0
GROUP_END = 7
if __name__ == '__main__':
    for i in range(GROUP_START, GROUP_END + 1):
        main(offset=i * 20 + 18)
        time.sleep(1)