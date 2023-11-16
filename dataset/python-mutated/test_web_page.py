import hashlib
from unittest.mock import Mock, patch
import pytest
from embedchain.loaders.web_page import WebPageLoader

@pytest.fixture
def web_page_loader():
    if False:
        return 10
    return WebPageLoader()

def test_load_data(web_page_loader):
    if False:
        while True:
            i = 10
    page_url = 'https://example.com/page'
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = '\n        <html>\n            <head>\n                <title>Test Page</title>\n            </head>\n            <body>\n                <div id="content">\n                    <p>This is some test content.</p>\n                </div>\n            </body>\n        </html>\n    '
    with patch('embedchain.loaders.web_page.requests.get', return_value=mock_response):
        result = web_page_loader.load_data(page_url)
    content = web_page_loader._get_clean_content(mock_response.content, page_url)
    expected_doc_id = hashlib.sha256((content + page_url).encode()).hexdigest()
    assert result['doc_id'] == expected_doc_id
    expected_data = [{'content': content, 'meta_data': {'url': page_url}}]
    assert result['data'] == expected_data

def test_get_clean_content_excludes_unnecessary_info(web_page_loader):
    if False:
        print('Hello World!')
    mock_html = '\n        <html>\n        <head>\n            <title>Sample HTML</title>\n            <style>\n                /* Stylesheet to be excluded */\n                .elementor-location-header {\n                    background-color: #f0f0f0;\n                }\n            </style>\n        </head>\n        <body>\n            <header id="header">Header Content</header>\n            <nav class="nav">Nav Content</nav>\n            <aside>Aside Content</aside>\n            <form>Form Content</form>\n            <main>Main Content</main>\n            <footer class="footer">Footer Content</footer>\n            <script>Some Script</script>\n            <noscript>NoScript Content</noscript>\n            <svg>SVG Content</svg>\n            <canvas>Canvas Content</canvas>\n            \n            <div id="sidebar">Sidebar Content</div>\n            <div id="main-navigation">Main Navigation Content</div>\n            <div id="menu-main-menu">Menu Main Menu Content</div>\n            \n            <div class="header-sidebar-wrapper">Header Sidebar Wrapper Content</div>\n            <div class="blog-sidebar-wrapper">Blog Sidebar Wrapper Content</div>\n            <div class="related-posts">Related Posts Content</div>\n        </body>\n        </html>\n    '
    tags_to_exclude = ['nav', 'aside', 'form', 'header', 'noscript', 'svg', 'canvas', 'footer', 'script', 'style']
    ids_to_exclude = ['sidebar', 'main-navigation', 'menu-main-menu']
    classes_to_exclude = ['elementor-location-header', 'navbar-header', 'nav', 'header-sidebar-wrapper', 'blog-sidebar-wrapper', 'related-posts']
    content = web_page_loader._get_clean_content(mock_html, 'https://example.com/page')
    for tag in tags_to_exclude:
        assert tag not in content
    for id in ids_to_exclude:
        assert id not in content
    for class_name in classes_to_exclude:
        assert class_name not in content
    assert len(content) > 0