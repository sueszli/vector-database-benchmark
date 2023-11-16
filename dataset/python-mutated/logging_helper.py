import logging.handlers

class TestHandler(logging.handlers.BufferingHandler):

    def __init__(self, matcher):
        if False:
            return 10
        logging.handlers.BufferingHandler.__init__(self, 0)
        self.matcher = matcher

    def shouldFlush(self):
        if False:
            print('Hello World!')
        return False

    def emit(self, record):
        if False:
            i = 10
            return i + 15
        self.format(record)
        self.buffer.append(record.__dict__)

    def matches(self, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Look for a saved dict whose keys/values match the supplied arguments.\n        '
        result = False
        for d in self.buffer:
            if self.matcher.matches(d, **kwargs):
                result = True
                break
        return result