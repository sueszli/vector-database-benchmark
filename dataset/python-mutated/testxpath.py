"""
Created on 2023/02/21
@author: Irony
@site: https://pyqt.site https://github.com/PyQt5
@email: 892768447@qq.com
@file: testxpath.py
@description:
"""
import os
from lxml.etree import HTML
Actor = '<a href="{href}" target="_blank" title="{title}" style="text-decoration: none;font-size: 12px;color: #999999;">{title}</a>&nbsp;'

def _makeItem(lis):
    if False:
        print('Hello World!')
    for li in lis:
        a = li.find('.//div/a')
        play_url = 'https://music.163.com' + a.get('href')
        img = li.find('.//div/img')
        cover_url = img.get('src')
        playlist_title = a.get('title')
        figure_info = 'aaa'
        figure_score = ''
        figure = li.xpath('.//p[2]/a')[0]
        playlist_author = '<span style="font-size: 12px;"作者：{}</span>'.format(Actor.format(href='https://music.163.com' + figure.get('href'), title=figure.get('title')))
        play_count = (li.xpath('.//div/div/span[2]/text()') or [''])[0]
        path = 'cache/{0}.jpg'.format(os.path.splitext(os.path.basename(cover_url).split('?')[0])[0])
        cover_path = 'Data/pic_v.png'
        if os.path.isfile(path):
            cover_path = path
        print(cover_path, playlist_title, playlist_author, play_count, play_url, cover_url, path)

def _parseHtml(html):
    if False:
        while True:
            i = 10
    html = HTML(html)
    lis = html.xpath("//ul[@id='m-pl-container']/li")
    _makeItem(lis)
if __name__ == '__main__':
    data = open('D:\\Computer\\Desktop\\163.html', 'rb').read()
    _parseHtml(data)