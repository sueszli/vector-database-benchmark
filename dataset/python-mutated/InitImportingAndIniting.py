from robot.libraries.BuiltIn import BuiltIn
from robot.api import logger

class Importing:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        BuiltIn().import_library('String')

    def kw_from_lib_with_importing_init(self):
        if False:
            i = 10
            return i + 15
        print('Keyword from library with importing __init__.')

class Initting:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.lib = BuiltIn().get_library_instance('InitImportingAndIniting.Initted')

    def kw_from_lib_with_initting_init(self):
        if False:
            i = 10
            return i + 15
        logger.info('Keyword from library with initting __init__.')
        self.lib.kw_from_lib_initted_by_init()

class Initted:

    def __init__(self, id):
        if False:
            for i in range(10):
                print('nop')
        self.id = id

    def kw_from_lib_initted_by_init(self):
        if False:
            print('Hello World!')
        print('Keyword from library initted by __init__ (id: %s).' % self.id)