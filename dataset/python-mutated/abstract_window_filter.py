import re
import typing

class AbstractWindowFilter:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.windowInfoRegex = None
        self.isRecursive = False

    def get_serializable(self):
        if False:
            print('Hello World!')
        if self.windowInfoRegex is not None:
            return {'regex': self.windowInfoRegex.pattern, 'isRecursive': self.isRecursive}
        else:
            return {'regex': None, 'isRecursive': False}

    def load_from_serialized(self, data):
        if False:
            while True:
                i = 10
        try:
            if isinstance(data, dict):
                self.set_window_titles(data['regex'])
                self.isRecursive = data['isRecursive']
            else:
                self.set_window_titles(data)
        except re.error as e:
            raise e

    def copy_window_filter(self, window_filter):
        if False:
            print('Hello World!')
        self.windowInfoRegex = window_filter.windowInfoRegex
        self.isRecursive = window_filter.isRecursive

    def set_window_titles(self, regex):
        if False:
            print('Hello World!')
        if regex is not None:
            try:
                self.windowInfoRegex = re.compile(regex, re.UNICODE)
            except re.error as e:
                raise e
        else:
            self.windowInfoRegex = regex

    def set_filter_recursive(self, recurse):
        if False:
            for i in range(10):
                print('nop')
        self.isRecursive = recurse

    def has_filter(self) -> bool:
        if False:
            return 10
        return self.windowInfoRegex is not None

    def inherits_filter(self) -> bool:
        if False:
            return 10
        if self.parent is not None:
            return self.parent.get_applicable_regex(True) is not None
        return False

    def get_child_filter(self):
        if False:
            print('Hello World!')
        if self.isRecursive and self.windowInfoRegex is not None:
            return self.get_filter_regex()
        elif self.parent is not None:
            return self.parent.get_child_filter()
        else:
            return ''

    def get_filter_regex(self):
        if False:
            while True:
                i = 10
        '\n        Used by the GUI to obtain human-readable version of the filter\n        '
        if self.windowInfoRegex is not None:
            if self.isRecursive:
                return self.windowInfoRegex.pattern
            else:
                return self.windowInfoRegex.pattern
        elif self.parent is not None:
            return self.parent.get_child_filter()
        else:
            return ''

    def filter_matches(self, otherFilter):
        if False:
            print('Hello World!')
        if otherFilter is None or self.get_applicable_regex() is None:
            return True
        return otherFilter == self.get_applicable_regex().pattern

    def same_filter_as_item(self, otherItem):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(otherItem, AbstractWindowFilter):
            return False
        return self.filter_matches(otherItem.get_applicable_regex)

    def get_applicable_regex(self, forChild=False):
        if False:
            while True:
                i = 10
        if self.windowInfoRegex is not None:
            if forChild and self.isRecursive or not forChild:
                return self.windowInfoRegex
        elif self.parent is not None:
            return self.parent.get_applicable_regex(True)
        return None

    def _should_trigger_window_title(self, window_info):
        if False:
            i = 10
            return i + 15
        r = self.get_applicable_regex()
        if r is not None:
            return bool(r.match(window_info.wm_title)) or bool(r.match(window_info.wm_class))
        else:
            return True