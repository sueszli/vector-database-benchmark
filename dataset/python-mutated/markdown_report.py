import time
import sys
from lib.core.settings import NEW_LINE
from lib.reports.base import FileBaseReport

class MarkdownReport(FileBaseReport):

    def get_header(self):
        if False:
            return 10
        header = '### Information' + NEW_LINE
        header += f'Command: {chr(32).join(sys.argv)}'
        header += NEW_LINE
        header += f'Time: {time.ctime()}'
        header += NEW_LINE * 2
        header += 'URL | Status | Size | Content Type | Redirection' + NEW_LINE
        header += '----|--------|------|--------------|------------' + NEW_LINE
        return header

    def generate(self, entries):
        if False:
            i = 10
            return i + 15
        output = self.get_header()
        for entry in entries:
            output += f'{entry.url} | {entry.status} | {entry.length} | {entry.type} | {entry.redirect}' + NEW_LINE
        return output