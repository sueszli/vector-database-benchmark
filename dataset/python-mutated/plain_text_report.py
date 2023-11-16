import time
import sys
from lib.core.settings import NEW_LINE
from lib.reports.base import FileBaseReport
from lib.utils.common import human_size

class PlainTextReport(FileBaseReport):

    def get_header(self):
        if False:
            return 10
        return f'# Dirsearch started {time.ctime()} as: {chr(32).join(sys.argv)}' + NEW_LINE * 2

    def generate(self, entries):
        if False:
            while True:
                i = 10
        output = self.get_header()
        for entry in entries:
            readable_size = human_size(entry.length)
            output += f'{entry.status}  {readable_size.rjust(6, chr(32))}  {entry.url}'
            if entry.redirect:
                output += f'    -> REDIRECTS TO: {entry.redirect}'
            output += NEW_LINE
        return output