import re
from collections.abc import Sequence
from typing import NamedTuple
from mitmproxy import ctx
from mitmproxy import exceptions
from mitmproxy import flowfilter
from mitmproxy import http
from mitmproxy.utils.spec import parse_spec

class MapRemoteSpec(NamedTuple):
    matches: flowfilter.TFilter
    subject: str
    replacement: str

def parse_map_remote_spec(option: str) -> MapRemoteSpec:
    if False:
        return 10
    spec = MapRemoteSpec(*parse_spec(option))
    try:
        re.compile(spec.subject)
    except re.error as e:
        raise ValueError(f'Invalid regular expression {spec.subject!r} ({e})')
    return spec

class MapRemote:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.replacements: list[MapRemoteSpec] = []

    def load(self, loader):
        if False:
            for i in range(10):
                print('nop')
        loader.add_option('map_remote', Sequence[str], [], '\n            Map remote resources to another remote URL using a pattern of the form\n            "[/flow-filter]/url-regex/replacement", where the separator can\n            be any character.\n            ')

    def configure(self, updated):
        if False:
            return 10
        if 'map_remote' in updated:
            self.replacements = []
            for option in ctx.options.map_remote:
                try:
                    spec = parse_map_remote_spec(option)
                except ValueError as e:
                    raise exceptions.OptionsError(f'Cannot parse map_remote option {option}: {e}') from e
                self.replacements.append(spec)

    def request(self, flow: http.HTTPFlow) -> None:
        if False:
            print('Hello World!')
        if flow.response or flow.error or (not flow.live):
            return
        for spec in self.replacements:
            if spec.matches(flow):
                url = flow.request.pretty_url
                new_url = re.sub(spec.subject, spec.replacement, url)
                if url != new_url:
                    flow.request.url = new_url