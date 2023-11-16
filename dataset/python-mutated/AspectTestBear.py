from coalib.bearlib.aspects import Root
from coalib.bears.LocalBear import LocalBear
from coalib.results.Result import Result
from coalib.results.RESULT_SEVERITY import RESULT_SEVERITY

class AspectTestBear(LocalBear, aspects={'detect': [Root.Redundancy.UnusedVariable.UnusedGlobalVariable, Root.Redundancy.UnusedVariable.UnusedLocalVariable]}, languages=['Python']):
    LANGUAGES = {'Python'}
    LICENSE = 'AGPL-3.0'

    def run(self, filename, file, config: str=''):
        if False:
            i = 10
            return i + 15
        '\n        Bear that have aspect.\n\n        :param config: An optional dummy config file.\n        '
        yield Result.from_values(origin=self, message='This is just a dummy result', severity=RESULT_SEVERITY.INFO, file=filename, aspect=Root.Redundancy.UnusedVariable.UnusedLocalVariable('py'))