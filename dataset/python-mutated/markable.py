class Markable:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__marked = set()
        self.__inverted = False

    def _did_mark(self, o):
        if False:
            return 10
        pass

    def _did_unmark(self, o):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _get_markable_count(self):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def _is_markable(self, o):
        if False:
            for i in range(10):
                print('nop')
        return True

    def _remove_mark_flag(self, o):
        if False:
            i = 10
            return i + 15
        try:
            self.__marked.remove(o)
            self._did_unmark(o)
        except KeyError:
            pass

    def is_marked(self, o):
        if False:
            return 10
        if not self._is_markable(o):
            return False
        is_marked = o in self.__marked
        if self.__inverted:
            is_marked = not is_marked
        return is_marked

    def mark(self, o):
        if False:
            i = 10
            return i + 15
        if self.is_marked(o):
            return False
        if not self._is_markable(o):
            return False
        return self.mark_toggle(o)

    def mark_multiple(self, objects):
        if False:
            i = 10
            return i + 15
        for o in objects:
            self.mark(o)

    def mark_all(self):
        if False:
            for i in range(10):
                print('nop')
        self.mark_none()
        self.__inverted = True

    def mark_invert(self):
        if False:
            return 10
        self.__inverted = not self.__inverted

    def mark_none(self):
        if False:
            return 10
        for o in self.__marked:
            self._did_unmark(o)
        self.__marked = set()
        self.__inverted = False

    def mark_toggle(self, o):
        if False:
            while True:
                i = 10
        try:
            self.__marked.remove(o)
            self._did_unmark(o)
        except KeyError:
            if not self._is_markable(o):
                return False
            self.__marked.add(o)
            self._did_mark(o)
        return True

    def mark_toggle_multiple(self, objects):
        if False:
            return 10
        for o in objects:
            self.mark_toggle(o)

    def unmark(self, o):
        if False:
            i = 10
            return i + 15
        if not self.is_marked(o):
            return False
        return self.mark_toggle(o)

    def unmark_multiple(self, objects):
        if False:
            for i in range(10):
                print('nop')
        for o in objects:
            self.unmark(o)

    @property
    def mark_count(self):
        if False:
            print('Hello World!')
        if self.__inverted:
            return self._get_markable_count() - len(self.__marked)
        else:
            return len(self.__marked)

    @property
    def mark_inverted(self):
        if False:
            print('Hello World!')
        return self.__inverted

class MarkableList(list, Markable):

    def __init__(self):
        if False:
            return 10
        list.__init__(self)
        Markable.__init__(self)

    def _get_markable_count(self):
        if False:
            return 10
        return len(self)

    def _is_markable(self, o):
        if False:
            for i in range(10):
                print('nop')
        return o in self