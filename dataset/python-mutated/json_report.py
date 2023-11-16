import json
import time
import sys
from lib.reports.base import FileBaseReport

class JSONReport(FileBaseReport):

    def generate(self, entries):
        if False:
            while True:
                i = 10
        report = {'info': {'args': ' '.join(sys.argv), 'time': time.ctime()}, 'results': []}
        for entry in entries:
            result = {'url': entry.url, 'status': entry.status, 'content-length': entry.length, 'content-type': entry.type, 'redirect': entry.redirect}
            report['results'].append(result)
        return json.dumps(report, sort_keys=True, indent=4)