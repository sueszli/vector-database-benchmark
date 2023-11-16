import time
from coalib.bears.LocalBear import LocalBear
from coalib.results.Result import Result

class ProcessingLocalTestRawFileBear(LocalBear):
    USE_RAW_FILES = True

    def run(self, filename, file):
        if False:
            for i in range(10):
                print('nop')
        time.sleep(0.05)
        return [Result('LocalTestRawBear', 'test msg')]