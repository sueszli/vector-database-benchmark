import logging
from functools import lru_cache
from ulauncher.modes.apps.AppMode import AppMode
from ulauncher.modes.calc.CalcMode import CalcMode
from ulauncher.modes.extensions.ExtensionMode import ExtensionMode
from ulauncher.modes.file_browser.FileBrowserMode import FileBrowserMode
from ulauncher.modes.shortcuts.ShortcutMode import ShortcutMode
logger = logging.getLogger()

class ModeHandler:

    @classmethod
    @lru_cache(maxsize=None)
    def get_instance(cls):
        if False:
            print('Hello World!')
        return cls([FileBrowserMode(), CalcMode(), ShortcutMode(), ExtensionMode(), AppMode()])

    def __init__(self, modes):
        if False:
            print('Hello World!')
        self.modes = modes

    def on_query_change(self, query):
        if False:
            print('Hello World!')
        '\n        Iterate over all search modes and run first enabled.\n        '
        for mode in self.modes:
            mode.on_query_change(query)
        mode = self.get_mode_from_query(query)
        if mode:
            return mode.handle_query(query)
        results = self.search(query)
        if not results and query:
            for mode in self.modes:
                results.extend(mode.get_fallback_results())
        return results

    def on_query_backspace(self, query):
        if False:
            for i in range(10):
                print('nop')
        mode = self.get_mode_from_query(query)
        return mode and mode.on_query_backspace(query)

    def get_mode_from_query(self, query):
        if False:
            while True:
                i = 10
        for mode in self.modes:
            if mode.is_enabled(query):
                return mode
        return None

    def search(self, query, min_score=50, limit=50):
        if False:
            i = 10
            return i + 15
        searchables = []
        for mode in self.modes:
            searchables.extend(list(mode.get_triggers()))
        sorted_ = sorted(searchables, key=lambda i: i.search_score(query), reverse=True)[:limit]
        return list(filter(lambda searchable: searchable.search_score(query) > min_score, sorted_))