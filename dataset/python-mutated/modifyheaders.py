import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple
from mitmproxy import ctx
from mitmproxy import exceptions
from mitmproxy import flowfilter
from mitmproxy import http
from mitmproxy.http import Headers
from mitmproxy.utils import strutils
from mitmproxy.utils.spec import parse_spec

class ModifySpec(NamedTuple):
    matches: flowfilter.TFilter
    subject: bytes
    replacement_str: str

    def read_replacement(self) -> bytes:
        if False:
            print('Hello World!')
        '\n        Process the replacement str. This usually just involves converting it to bytes.\n        However, if it starts with `@`, we interpret the rest as a file path to read from.\n\n        Raises:\n            - IOError if the file cannot be read.\n        '
        if self.replacement_str.startswith('@'):
            return Path(self.replacement_str[1:]).expanduser().read_bytes()
        else:
            return strutils.escaped_str_to_bytes(self.replacement_str)

def parse_modify_spec(option: str, subject_is_regex: bool) -> ModifySpec:
    if False:
        while True:
            i = 10
    (flow_filter, subject_str, replacement) = parse_spec(option)
    subject = strutils.escaped_str_to_bytes(subject_str)
    if subject_is_regex:
        try:
            re.compile(subject)
        except re.error as e:
            raise ValueError(f'Invalid regular expression {subject!r} ({e})')
    spec = ModifySpec(flow_filter, subject, replacement)
    try:
        spec.read_replacement()
    except OSError as e:
        raise ValueError(f'Invalid file path: {replacement[1:]} ({e})')
    return spec

class ModifyHeaders:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.replacements: list[ModifySpec] = []

    def load(self, loader):
        if False:
            print('Hello World!')
        loader.add_option('modify_headers', Sequence[str], [], '\n            Header modify pattern of the form "[/flow-filter]/header-name/[@]header-value", where the\n            separator can be any character. The @ allows to provide a file path that is used to read\n            the header value string. An empty header-value removes existing header-name headers.\n            ')

    def configure(self, updated):
        if False:
            while True:
                i = 10
        if 'modify_headers' in updated:
            self.replacements = []
            for option in ctx.options.modify_headers:
                try:
                    spec = parse_modify_spec(option, False)
                except ValueError as e:
                    raise exceptions.OptionsError(f'Cannot parse modify_headers option {option}: {e}') from e
                self.replacements.append(spec)

    def request(self, flow):
        if False:
            return 10
        if flow.response or flow.error or (not flow.live):
            return
        self.run(flow, flow.request.headers)

    def response(self, flow):
        if False:
            return 10
        if flow.error or not flow.live:
            return
        self.run(flow, flow.response.headers)

    def run(self, flow: http.HTTPFlow, hdrs: Headers) -> None:
        if False:
            print('Hello World!')
        matches = []
        for spec in self.replacements:
            matches.append(spec.matches(flow))
        for (i, spec) in enumerate(self.replacements):
            if matches[i]:
                hdrs.pop(spec.subject, None)
        for (i, spec) in enumerate(self.replacements):
            if matches[i]:
                try:
                    replacement = spec.read_replacement()
                except OSError as e:
                    logging.warning(f'Could not read replacement file: {e}')
                    continue
                else:
                    if replacement:
                        hdrs.add(spec.subject, replacement)