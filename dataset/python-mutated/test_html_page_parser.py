from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from poetry.repositories.parsers.html_page_parser import HTMLPageParser
if TYPE_CHECKING:
    from tests.types import HTMLPageGetter

@pytest.fixture()
def html_page(html_page_content: HTMLPageGetter) -> str:
    if False:
        for i in range(10):
            print('nop')
    links = '\n        <a href="https://example.org/demo-0.1.whl">demo-0.1.whl</a><br/>\n        <a href="https://example.org/demo-0.1.whl"\n            data-requires-python=">=3.7">demo-0.1.whl</a><br/>\n        <a href="https://example.org/demo-0.1.whl" data-yanked>demo-0.1.whl</a><br/>\n        <a href="https://example.org/demo-0.1.whl" data-yanked="">demo-0.1.whl</a><br/>\n        <a href="https://example.org/demo-0.1.whl"\n            data-yanked="<reason>"\n        >demo-0.1.whl</a><br/>\n        <a href="https://example.org/demo-0.1.whl"\n            data-requires-python=">=3.7"\n            data-yanked\n         >demo-0.1.whl</a><br/>\n    '
    return html_page_content(links)

def test_html_page_parser_anchors(html_page: str) -> None:
    if False:
        i = 10
        return i + 15
    parser = HTMLPageParser()
    parser.feed(html_page)
    assert parser.anchors == [{'href': 'https://example.org/demo-0.1.whl'}, {'data-requires-python': '>=3.7', 'href': 'https://example.org/demo-0.1.whl'}, {'data-yanked': None, 'href': 'https://example.org/demo-0.1.whl'}, {'data-yanked': '', 'href': 'https://example.org/demo-0.1.whl'}, {'data-yanked': '<reason>', 'href': 'https://example.org/demo-0.1.whl'}, {'data-requires-python': '>=3.7', 'data-yanked': None, 'href': 'https://example.org/demo-0.1.whl'}]

def test_html_page_parser_base_url() -> None:
    if False:
        print('Hello World!')
    content = '\n        <!DOCTYPE html>\n        <html>\n          <head>\n            <base href="https://example.org/">\n            <meta name="pypi:repository-version" content="1.0">\n            <title>Links for demo</title>\n          </head>\n          <body>\n            <h1>Links for demo</h1>\n            <a href="demo-0.1.whl">demo-0.1.whl</a><br/>\n            </body>\n        </html>\n    '
    parser = HTMLPageParser()
    parser.feed(content)
    assert parser.base_url == 'https://example.org/'