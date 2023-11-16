"""
Topic: 解析简单的XML
Desc : 
"""
from urllib.request import urlopen
from xml.etree.ElementTree import parse

def simple_xml():
    if False:
        print('Hello World!')
    u = urlopen('http://planet.python.org/rss20.xml')
    doc = parse(u)
    for item in doc.iterfind('channel/item'):
        title = item.findtext('title')
        date = item.findtext('pubDate')
        link = item.findtext('link')
        print(title)
        print(date)
        print(link)
        print()
if __name__ == '__main__':
    simple_xml()