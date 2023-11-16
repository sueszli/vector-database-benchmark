"""Preprocess function to filter/prepare Wikipedia docs."""
import regex as re
from html.parser import HTMLParser
PARSER = HTMLParser()
BLACKLIST = set(['23443579', '52643645'])

def preprocess(article):
    if False:
        return 10
    for (k, v) in article.items():
        article[k] = PARSER.unescape(v)
    if article['id'] in BLACKLIST:
        return None
    if '(disambiguation)' in article['title'].lower():
        return None
    if '(disambiguation page)' in article['title'].lower():
        return None
    if re.match('(List of .+)|(Index of .+)|(Outline of .+)', article['title']):
        return None
    return {'id': article['title'], 'text': article['text']}