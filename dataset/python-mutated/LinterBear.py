from coalib.bearlib.abstractions.Linter import linter

@linter(executable='echo', output_format='regex', output_regex='.+:(?P<line>\\d+):(?P<message>.*)')
class EchoBear:
    CAN_DETECT = {'Syntax', 'Security'}
    CAN_FIX = {'Redundancy'}
    '\n    A simple bear to test that collectors are importing also bears that are\n    defined in another file *but* have baseclasses in the right file.\n\n    (linter will create a new class that inherits from this class.)\n    '

    @staticmethod
    def create_arguments(filename, file, config_file):
        if False:
            for i in range(10):
                print('nop')
        return ()