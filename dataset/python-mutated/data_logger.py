"""Log data to a jsonl file."""
import datetime
import json
import os
import time
from typing import Any, Dict, Text
from open_spiel.python.utils import gfile

class DataLoggerJsonLines:
    """Log data to a jsonl file."""

    def __init__(self, path: str, name: str, flush=True):
        if False:
            print('Hello World!')
        self._fd = gfile.Open(os.path.join(path, name + '.jsonl'), 'w')
        self._flush = flush
        self._start_time = time.time()

    def __del__(self):
        if False:
            while True:
                i = 10
        self.close()

    def close(self):
        if False:
            return 10
        if hasattr(self, '_fd') and self._fd is not None:
            self._fd.flush()
            self._fd.close()
            self._fd = None

    def flush(self):
        if False:
            print('Hello World!')
        self._fd.flush()

    def write(self, data: Dict[Text, Any]):
        if False:
            print('Hello World!')
        now = time.time()
        data['time_abs'] = now
        data['time_rel'] = now - self._start_time
        dt_now = datetime.datetime.utcfromtimestamp(now)
        data['time_str'] = dt_now.strftime('%Y-%m-%d %H:%M:%S.%f +0000')
        self._fd.write(json.dumps(data))
        self._fd.write('\n')
        if self._flush:
            self.flush()