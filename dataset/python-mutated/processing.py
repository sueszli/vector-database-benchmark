import re
from typing import Optional, List
from ..plugins import ConverterPlugin
from ..plugins.registry import plugin_manager
from ..context import Environment
MIME_RE = re.compile('^[^/]+/[^/]+$')

def is_valid_mime(mime):
    if False:
        while True:
            i = 10
    return mime and MIME_RE.match(mime)

class Conversion:

    @staticmethod
    def get_converter(mime: str) -> Optional[ConverterPlugin]:
        if False:
            i = 10
            return i + 15
        if is_valid_mime(mime):
            for converter_class in plugin_manager.get_converters():
                if converter_class.supports(mime):
                    return converter_class(mime)

class Formatting:
    """A delegate class that invokes the actual processors."""

    def __init__(self, groups: List[str], env=Environment(), **kwargs):
        if False:
            print('Hello World!')
        '\n        :param groups: names of processor groups to be applied\n        :param env: Environment\n        :param kwargs: additional keyword arguments for processors\n\n        '
        available_plugins = plugin_manager.get_formatters_grouped()
        self.enabled_plugins = []
        for group in groups:
            for cls in available_plugins[group]:
                p = cls(env=env, **kwargs)
                if p.enabled:
                    self.enabled_plugins.append(p)

    def format_headers(self, headers: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        for p in self.enabled_plugins:
            headers = p.format_headers(headers)
        return headers

    def format_body(self, content: str, mime: str) -> str:
        if False:
            i = 10
            return i + 15
        if is_valid_mime(mime):
            for p in self.enabled_plugins:
                content = p.format_body(content, mime)
        return content

    def format_metadata(self, metadata: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        for p in self.enabled_plugins:
            metadata = p.format_metadata(metadata)
        return metadata