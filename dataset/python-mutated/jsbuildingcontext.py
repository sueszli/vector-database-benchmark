from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from robot.output.loggerhelper import LEVELS
from robot.utils import attribute_escape, get_link_path, html_escape, safe_str
from .expandkeywordmatcher import ExpandKeywordMatcher
from .stringcache import StringCache

class JsBuildingContext:

    def __init__(self, log_path=None, split_log=False, expand_keywords=None, prune_input=False):
        if False:
            for i in range(10):
                print('nop')
        self._log_dir = self._get_log_dir(log_path)
        self._split_log = split_log
        self._prune_input = prune_input
        self._strings = self._top_level_strings = StringCache()
        self.basemillis = None
        self.split_results = []
        self.min_level = 'NONE'
        self._msg_links = {}
        self._expand_matcher = ExpandKeywordMatcher(expand_keywords) if expand_keywords else None

    def _get_log_dir(self, log_path):
        if False:
            print('Hello World!')
        if isinstance(log_path, Path):
            return log_path.parent
        if isinstance(log_path, str):
            return Path(log_path).parent
        return None

    def string(self, string, escape=True, attr=False):
        if False:
            for i in range(10):
                print('nop')
        if not string:
            return self._strings.empty
        if escape:
            if not isinstance(string, str):
                string = safe_str(string)
            string = (html_escape if not attr else attribute_escape)(string)
        return self._strings.add(string)

    def html(self, string):
        if False:
            while True:
                i = 10
        return self._strings.add(string, html=True)

    def relative_source(self, source):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(source, str):
            source = Path(source)
        rel_source = get_link_path(source, self._log_dir) if self._log_dir and source and source.exists() else ''
        return self.string(rel_source)

    def timestamp(self, ts: datetime) -> 'int|None':
        if False:
            i = 10
            return i + 15
        if not ts:
            return None
        millis = round(ts.timestamp() * 1000)
        if self.basemillis is None:
            self.basemillis = millis
        return millis - self.basemillis

    def message_level(self, level):
        if False:
            return 10
        if LEVELS[level] < LEVELS[self.min_level]:
            self.min_level = level

    def create_link_target(self, msg):
        if False:
            print('Hello World!')
        id = self._top_level_strings.add(msg.parent.id)
        self._msg_links[self._link_key(msg)] = id

    def check_expansion(self, kw):
        if False:
            while True:
                i = 10
        if self._expand_matcher is not None:
            self._expand_matcher.match(kw)

    @property
    def expand_keywords(self):
        if False:
            return 10
        return self._expand_matcher.matched_ids if self._expand_matcher else None

    def link(self, msg):
        if False:
            i = 10
            return i + 15
        return self._msg_links.get(self._link_key(msg))

    def _link_key(self, msg):
        if False:
            for i in range(10):
                print('nop')
        return (msg.message, msg.level, msg.timestamp)

    @property
    def strings(self):
        if False:
            for i in range(10):
                print('nop')
        return self._strings.dump()

    def start_splitting_if_needed(self, split=False):
        if False:
            i = 10
            return i + 15
        if self._split_log and split:
            self._strings = StringCache()
            return True
        return False

    def end_splitting(self, model):
        if False:
            while True:
                i = 10
        self.split_results.append((model, self.strings))
        self._strings = self._top_level_strings
        return len(self.split_results)

    @contextmanager
    def prune_input(self, *items):
        if False:
            i = 10
            return i + 15
        yield
        if self._prune_input:
            for item in items:
                item.clear()