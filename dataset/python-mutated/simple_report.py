from lib.core.settings import NEW_LINE
from lib.reports.base import FileBaseReport

class SimpleReport(FileBaseReport):

    def generate(self, entries):
        if False:
            i = 10
            return i + 15
        return NEW_LINE.join((entry.url for entry in entries))