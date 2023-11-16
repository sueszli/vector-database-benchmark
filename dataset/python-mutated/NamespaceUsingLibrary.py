from robot.libraries.BuiltIn import BuiltIn

class NamespaceUsingLibrary:

    def __init__(self):
        if False:
            print('Hello World!')
        self._importing_suite = BuiltIn().get_variable_value('${SUITE NAME}')
        self._easter = BuiltIn().get_library_instance('Easter')

    def get_importing_suite(self):
        if False:
            while True:
                i = 10
        return self._importing_suite

    def get_other_library(self):
        if False:
            return 10
        return self._easter