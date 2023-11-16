import pytest
import responses
from bs4 import BeautifulSoup

@pytest.mark.parametrize('ignored_tag', ['<nav>This is a navigation bar.</nav>', '<aside>This is an aside.</aside>', '<form>This is a form.</form>', '<header>This is a header.</header>', '<noscript>This is a noscript.</noscript>', '<svg>This is an SVG.</svg>', '<canvas>This is a canvas.</canvas>', '<footer>This is a footer.</footer>', '<script>This is a script.</script>', '<style>This is a style.</style>'], ids=['nav', 'aside', 'form', 'header', 'noscript', 'svg', 'canvas', 'footer', 'script', 'style'])
@pytest.mark.parametrize('selectee', ['\n<article class="bd-article">\n    <h2>Article Title</h2>\n    <p>Article content goes here.</p>\n    {ignored_tag}\n</article>\n', '\n<article role="main">\n    <h2>Main Article Title</h2>\n    <p>Main article content goes here.</p>\n    {ignored_tag}\n</article>\n', '\n<div class="md-content">\n    <h2>Markdown Content</h2>\n    <p>Markdown content goes here.</p>\n    {ignored_tag}\n</div>\n', '\n<div role="main">\n    <h2>Main Content</h2>\n    <p>Main content goes here.</p>\n    {ignored_tag}\n</div>\n', '\n<div class="container">\n    <h2>Container</h2>\n    <p>Container content goes here.</p>\n    {ignored_tag}\n</div>\n        ', '\n<div class="section">\n    <h2>Section</h2>\n    <p>Section content goes here.</p>\n    {ignored_tag}\n</div>\n        ', '\n<article>\n    <h2>Generic Article</h2>\n    <p>Generic article content goes here.</p>\n    {ignored_tag}\n</article>\n        ', '\n<main>\n    <h2>Main Content</h2>\n    <p>Main content goes here.</p>\n    {ignored_tag}\n</main>\n'], ids=['article.bd-article', 'article[role="main"]', 'div.md-content', 'div[role="main"]', 'div.container', 'div.section', 'article', 'main'])
def test_load_data_gets_by_selectors_and_ignored_tags(selectee, ignored_tag, loader, mocked_responses, mocker):
    if False:
        print('Hello World!')
    child_url = 'https://docs.embedchain.ai/quickstart'
    selectee = selectee.format(ignored_tag=ignored_tag)
    html_body = '\n<!DOCTYPE html>\n<html lang="en">\n<body>\n    {selectee}\n</body>\n</html>\n'
    html_body = html_body.format(selectee=selectee)
    mocked_responses.get(child_url, body=html_body, status=200, content_type='text/html')
    url = 'https://docs.embedchain.ai/'
    html_body = '\n<!DOCTYPE html>\n<html lang="en">\n<body>\n    <li><a href="/quickstart">Quickstart</a></li>\n</body>\n</html>\n'
    mocked_responses.get(url, body=html_body, status=200, content_type='text/html')
    mock_sha256 = mocker.patch('embedchain.loaders.docs_site_loader.hashlib.sha256')
    doc_id = 'mocked_hash'
    mock_sha256.return_value.hexdigest.return_value = doc_id
    result = loader.load_data(url)
    selector_soup = BeautifulSoup(selectee, 'html.parser')
    expected_content = ' '.join((selector_soup.select_one('h2').get_text(), selector_soup.select_one('p').get_text()))
    assert result['doc_id'] == doc_id
    assert result['data'] == [{'content': expected_content, 'meta_data': {'url': 'https://docs.embedchain.ai/quickstart'}}]

def test_load_data_gets_child_links_recursively(loader, mocked_responses, mocker):
    if False:
        i = 10
        return i + 15
    child_url = 'https://docs.embedchain.ai/quickstart'
    html_body = '\n<!DOCTYPE html>\n<html lang="en">\n<body>\n    <li><a href="/">..</a></li>\n    <li><a href="/quickstart">.</a></li>\n</body>\n</html>\n'
    mocked_responses.get(child_url, body=html_body, status=200, content_type='text/html')
    child_url = 'https://docs.embedchain.ai/introduction'
    html_body = '\n<!DOCTYPE html>\n<html lang="en">\n<body>\n    <li><a href="/">..</a></li>\n    <li><a href="/introduction">.</a></li>\n</body>\n</html>\n'
    mocked_responses.get(child_url, body=html_body, status=200, content_type='text/html')
    url = 'https://docs.embedchain.ai/'
    html_body = '\n<!DOCTYPE html>\n<html lang="en">\n<body>\n    <li><a href="/quickstart">Quickstart</a></li>\n    <li><a href="/introduction">Introduction</a></li>\n</body>\n</html>\n'
    mocked_responses.get(url, body=html_body, status=200, content_type='text/html')
    mock_sha256 = mocker.patch('embedchain.loaders.docs_site_loader.hashlib.sha256')
    doc_id = 'mocked_hash'
    mock_sha256.return_value.hexdigest.return_value = doc_id
    result = loader.load_data(url)
    assert result['doc_id'] == doc_id
    expected_data = [{'content': '..\n.', 'meta_data': {'url': 'https://docs.embedchain.ai/quickstart'}}, {'content': '..\n.', 'meta_data': {'url': 'https://docs.embedchain.ai/introduction'}}]
    assert all((item in expected_data for item in result['data']))

def test_load_data_fails_to_fetch_website(loader, mocked_responses, mocker):
    if False:
        i = 10
        return i + 15
    child_url = 'https://docs.embedchain.ai/introduction'
    mocked_responses.get(child_url, status=404)
    url = 'https://docs.embedchain.ai/'
    html_body = '\n<!DOCTYPE html>\n<html lang="en">\n<body>\n    <li><a href="/introduction">Introduction</a></li>\n</body>\n</html>\n'
    mocked_responses.get(url, body=html_body, status=200, content_type='text/html')
    mock_sha256 = mocker.patch('embedchain.loaders.docs_site_loader.hashlib.sha256')
    doc_id = 'mocked_hash'
    mock_sha256.return_value.hexdigest.return_value = doc_id
    result = loader.load_data(url)
    assert result['doc_id'] is doc_id
    assert result['data'] == []

@pytest.fixture
def loader():
    if False:
        for i in range(10):
            print('nop')
    from embedchain.loaders.docs_site_loader import DocsSiteLoader
    return DocsSiteLoader()

@pytest.fixture
def mocked_responses():
    if False:
        return 10
    with responses.RequestsMock() as rsps:
        yield rsps