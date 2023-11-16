import b2.build.type as type
type.register('VERBATIM', ['verbatim'])
import b2.build.scanner as scanner

class VerbatimScanner(scanner.CommonScanner):

    def pattern(self):
        if False:
            return 10
        return '//###include[ ]*"([^"]*)"'
scanner.register(VerbatimScanner, ['include'])
type.set_scanner('VERBATIM', VerbatimScanner)
import b2.build.generators as generators
generators.register_standard('verbatim.inline-file', ['VERBATIM'], ['CPP'])
from b2.manager import get_manager
get_manager().engine().register_action('verbatim.inline-file', '\n./inline_file.py $(<) $(>)\n')