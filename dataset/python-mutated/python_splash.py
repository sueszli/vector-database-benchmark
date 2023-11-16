"""
使用Splash服务器抓取Ajax渲染页面
"""
import json
import requests
CRAWLER_URL = 'http://weixin.sogou.com/weixin?page=1&type=2&query=%E4%B8%AD%E5%9B%BD'

def test_1(url):
    if False:
        for i in range(10):
            print('nop')
    render = 'http://xx.xx.xx.xx:8050/render.html'
    body = json.dumps({'url': url, 'wait': 0.5, 'images': 0, 'timeout': 3, 'allowed_content_types': 'text/html; charset=utf-8'})
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=render, headers=headers, data=body)
    print(url, response.status_code)
    print(response.text)
    return

def test_2(url):
    if False:
        for i in range(10):
            print('nop')
    render = 'http://xx.xx.xx.xx:8050/render.png?url=%s&timeout=5' % url
    response = requests.get(url=render)
    print(url, response.status_code)
    return