import logging
import re
from html.parser import HTMLParser
from ludwig.utils import strings_utils
logger = logging.getLogger(__name__)

class HTMLStripper(HTMLParser):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, data):
        if False:
            print('Hello World!')
        self.fed.append(data)

    def get_data(self):
        if False:
            i = 10
            return i + 15
        return ''.join(self.fed)

    def error(self, message):
        if False:
            return 10
        logger.error(message)

def strip_tags(html):
    if False:
        while True:
            i = 10
    stripper = HTMLStripper()
    stripper.feed(html)
    return stripper.get_data()
res_pre = [(re.compile('([^.:;\\?\\!>])(<br/?>)'), '\\1.\\2'), (re.compile('<br/?>'), ' ')]
res_post = [(re.compile('[\xa0\\t\\0]'), ' '), (re.compile('[–_]'), '-'), (re.compile('[\\’\\‘]'), "),\n    (re.compile(r'[”“]]'), r"), (re.compile('℅'), '%'), (re.compile('([^.>])(<br/?>)'), '\\1.\\2'), (re.compile('\\\\\\\\[NnRr]'), ' '), (re.compile('\\\\[NnRr]'), ' '), (re.compile('[\\n\\r]'), ' '), (re.compile('\\\\\\\\'), ' / '), (re.compile('<br/?>'), ' '), (re.compile('\\\\\\\\'), "\\'"), (re.compile("^\\'([^\\']+)$"), '\\1'), (re.compile("([\\<\\>\\{\\}\\[\\]\\(\\)\\-\\+\\=:;,\\./\\?\\!\\$%&£#@\\'₹ ])\\1+"), '\\1'), (re.compile("[^qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890\\<\\>\\{\\}\\[\\]\\(\\)\\-\\+\\=:;,\\./\\?\\!\\$%&£#@\\'₹ ]"), ' '), (re.compile('\\s{2,}'), ' ')]

def clean_html(html_text):
    if False:
        return 10
    (html_text, matched) = strings_utils.match_replace(html_text, res_pre)
    html_text = strip_tags(html_text)
    html_text = strings_utils.strip_accents(html_text)
    (html_text, matched) = strings_utils.match_replace(html_text, res_post)
    return html_text