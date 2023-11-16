from logging.handlers import BufferingHandler

class FTBufferingHandler(BufferingHandler):

    def flush(self):
        if False:
            i = 10
            return i + 15
        '\n        Override Flush behaviour - we keep half of the configured capacity\n        otherwise, we have moments with "empty" logs.\n        '
        self.acquire()
        try:
            self.buffer = self.buffer[-int(self.capacity / 2):]
        finally:
            self.release()