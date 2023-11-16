from typing import Mapping, Union
from bs4 import BeautifulSoup
from django.http import HttpRequest
from django.utils.html import escape
from zerver.lib.cache import cache_with_key, open_graph_description_cache_key

def html_to_text(content: Union[str, bytes], tags: Mapping[str, str]={'p': ' | '}) -> str:
    if False:
        for i in range(10):
            print('nop')
    bs = BeautifulSoup(content, features='lxml')
    for tag in bs.find_all('div', class_='admonition'):
        tag.clear()
    for tag in bs.find_all('div', class_='tabbed-section'):
        tag.clear()
    text = ''
    for element in bs.find_all(tags.keys()):
        if not element.text:
            continue
        if text:
            text += tags[element.name]
        text += element.text
        if len(text) > 500:
            break
    return escape(' '.join(text.split()))

@cache_with_key(open_graph_description_cache_key, timeout=3600 * 24)
def get_content_description(content: bytes, request: HttpRequest) -> str:
    if False:
        print('Hello World!')
    return html_to_text(content)