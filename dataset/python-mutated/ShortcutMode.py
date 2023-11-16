from ulauncher.modes.BaseMode import BaseMode
from ulauncher.modes.shortcuts.ShortcutResult import ShortcutResult
from ulauncher.modes.shortcuts.ShortcutsDb import ShortcutsDb

class ShortcutMode(BaseMode):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.shortcutsDb = ShortcutsDb.load()

    def is_enabled(self, query):
        if False:
            i = 10
            return i + 15
        '\n        Return True if mode should be enabled for a query\n        '
        return bool(self._get_active_shortcut(query))

    def _get_active_shortcut(self, query):
        if False:
            return 10
        for s in self.shortcutsDb.values():
            if query.startswith(f'{s.keyword} ') or (query == s.keyword and s.run_without_argument):
                return s
        return None

    def _create_items(self, shortcuts):
        if False:
            while True:
                i = 10
        return [ShortcutResult(**s) for s in shortcuts]

    def handle_query(self, query):
        if False:
            i = 10
            return i + 15
        '\n        @return Action object\n        '
        shortcut = self._get_active_shortcut(query)
        if not shortcut:
            msg = "Query doesn't match any shortcut"
            raise RuntimeError(msg)
        return [ShortcutResult(**shortcut)]

    def get_fallback_results(self):
        if False:
            while True:
                i = 10
        return self._create_items([s for s in self.shortcutsDb.values() if s['is_default_search']])

    def get_triggers(self):
        if False:
            return 10
        return self._create_items(list(self.shortcutsDb.values()))