from coalib.bears.LocalBear import LocalBear
from coalib.results.Result import Result
from coalib.results.RESULT_SEVERITY import RESULT_SEVERITY

class LineCountTestBear(LocalBear):
    LANGUAGES = {'all'}

    def run(self, filename, file):
        if False:
            while True:
                i = 10
        '\n        Counts the lines of each file.\n        '
        yield Result.from_values(origin=self, message=f'This file has {len(file)} lines.', severity=RESULT_SEVERITY.INFO, file=filename)