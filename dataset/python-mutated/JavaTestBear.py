from coalib.bears.LocalBear import LocalBear

class JavaTestBear(LocalBear):
    LANGUAGES = {'java'}
    LICENSE = 'AGPL-3.0'
    CAN_DETECT = {'Syntax', 'Formatting'}
    CAN_FIX = {'Security'}

    def run(self, filename, file, config: str=''):
        if False:
            while True:
                i = 10
        '\n        Bear to test that collecting of languages works.\n\n        :param config: An optional dummy config file.\n        '