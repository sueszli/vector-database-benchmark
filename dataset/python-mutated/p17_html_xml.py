"""
Topic: 处理html和xml文本
Desc : 
"""
import html

def html_xml():
    if False:
        while True:
            i = 10
    s = 'Elements are written as "<tag>text</tag>".'
    print(s)
    print(html.escape(s))
    print(html.escape(s, quote=False))
    s = 'Spicy Jalapeño'
    print(s.encode('ascii', errors='xmlcharrefreplace'))
if __name__ == '__main__':
    html_xml()