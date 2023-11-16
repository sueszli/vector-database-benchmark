from html.parser import HTMLParser
from typing import Dict, Iterable, List, NoReturn, Optional, Tuple

class TestHtmlParser(HTMLParser):
    """A generic HTML page parser which extracts useful things from the HTML"""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.links: List[str] = []
        self.hiddens: Dict[str, Optional[str]] = {}
        self.radios: Dict[str, List[Optional[str]]] = {}

    def handle_starttag(self, tag: str, attrs: Iterable[Tuple[str, Optional[str]]]) -> None:
        if False:
            return 10
        attr_dict = dict(attrs)
        if tag == 'a':
            href = attr_dict['href']
            if href:
                self.links.append(href)
        elif tag == 'input':
            input_name = attr_dict.get('name')
            if attr_dict['type'] == 'radio':
                assert input_name
                self.radios.setdefault(input_name, []).append(attr_dict['value'])
            elif attr_dict['type'] == 'hidden':
                assert input_name
                self.hiddens[input_name] = attr_dict['value']

    def error(self, message: str) -> NoReturn:
        if False:
            print('Hello World!')
        raise AssertionError(message)