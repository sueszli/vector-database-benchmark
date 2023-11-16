from collections import namedtuple
import pytest
from pyquery import PyQuery
from readthedocs.embed.utils import clean_references
URLData = namedtuple('URLData', ['docurl', 'ref', 'expected'])
html_base_url = 'https://t.readthedocs.io/en/latest/page.html'
dirhtml_base_url = 'https://t.readthedocs.io/en/latest/page/'
htmldata = [URLData(html_base_url, '#to-a-section', 'https://t.readthedocs.io/en/latest/page.html#to-a-section'), URLData(html_base_url, '/section.html', '/section.html'), URLData(html_base_url, 'internal/deep/section.html', 'https://t.readthedocs.io/en/latest/internal/deep/section.html'), URLData(html_base_url, 'section.html', 'https://t.readthedocs.io/en/latest/section.html'), URLData(html_base_url, 'relative/page.html#to-a-section', 'https://t.readthedocs.io/en/latest/relative/page.html#to-a-section'), URLData('https://t.readthedocs.io/en/latest/internal/deep/page/section.html', '../../page.html#to-a-section', 'https://t.readthedocs.io/en/latest/internal/deep/page/../../page.html#to-a-section'), URLData('https://t.readthedocs.io/en/latest/internal/deep/page/section.html', 'relative/page.html#to-a-section', 'https://t.readthedocs.io/en/latest/internal/deep/page/relative/page.html#to-a-section'), URLData(html_base_url, 'https://readthedocs.org/', 'https://readthedocs.org/')]
dirhtmldata = [URLData(dirhtml_base_url, '#to-a-section', 'https://t.readthedocs.io/en/latest/page/#to-a-section'), URLData(dirhtml_base_url, '/section/', '/section/'), URLData(dirhtml_base_url, 'internal/deep/section/', 'https://t.readthedocs.io/en/latest/page/internal/deep/section/'), URLData(dirhtml_base_url, 'section/', 'https://t.readthedocs.io/en/latest/page/section/'), URLData(dirhtml_base_url, 'relative/page/#to-a-section', 'https://t.readthedocs.io/en/latest/page/relative/page/#to-a-section'), URLData('https://t.readthedocs.io/en/latest/internal/deep/page/section/', '../../page/#to-a-section', 'https://t.readthedocs.io/en/latest/internal/deep/page/section/../../page/#to-a-section'), URLData(dirhtml_base_url, 'https://readthedocs.org/', 'https://readthedocs.org/')]
imagedata = [URLData(html_base_url, '/_images/image.png', '/_images/image.png'), URLData(html_base_url, 'relative/section/image.png', 'https://t.readthedocs.io/en/latest/relative/section/image.png'), URLData('https://t.readthedocs.io/en/latest/internal/deep/page/topic.html', '../../../_images/image.png', 'https://t.readthedocs.io/en/latest/internal/deep/page/../../../_images/image.png')]

@pytest.mark.parametrize('url', htmldata + dirhtmldata)
def test_clean_links(url):
    if False:
        for i in range(10):
            print('nop')
    pq = PyQuery(f'<body><a href="{url.ref}">Click here</a></body>')
    response = clean_references(pq, url.docurl)
    assert response.find('a').attr['href'] == url.expected

@pytest.mark.parametrize('url', imagedata)
def test_clean_images(url):
    if False:
        print('Hello World!')
    pq = PyQuery(f'<body><img alt="image alt content" src="{url.ref}"></img></body>')
    response = clean_references(pq, url.docurl)
    assert response.find('img').attr['src'] == url.expected

def test_two_links():
    if False:
        for i in range(10):
            print('nop')
    '\n    First link does not affect the second one.\n\n    We are using ``._replace`` for the firsturl case, and that should not affect\n    the second link.\n    '
    firsturl = URLData('https://t.readthedocs.io/en/latest/internal/deep/page/section.html', '../../page.html#to-a-section', 'https://t.readthedocs.io/en/latest/internal/deep/page/../../page.html#to-a-section')
    secondurl = URLData('', '#to-a-section', 'https://t.readthedocs.io/en/latest/internal/deep/page/section.html#to-a-section')
    pq = PyQuery(f'<body><a href="{firsturl.ref}">Click here</a><a href="{secondurl.ref}">Click here</a></body>')
    response = clean_references(pq, firsturl.docurl)
    (firstlink, secondlink) = response.find('a')
    assert (firstlink.attrib['href'], secondlink.attrib['href']) == (firsturl.expected, secondurl.expected)