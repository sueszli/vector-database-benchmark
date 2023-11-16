import warnings
from robot.utils import NormalizedDict

class _RunKeywordRegister:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._libs = {}

    def register_run_keyword(self, libname, keyword, args_to_process, deprecation_warning=True, dry_run=False):
        if False:
            while True:
                i = 10
        'Deprecated API for registering "run keyword variants".\n\n        Registered keywords are handled specially by Robot so that:\n\n        - Their arguments are not resolved normally (use ``args_to_process``\n          to control that). This mainly means replacing variables and handling\n          escapes.\n        - They are not stopped by timeouts.\n        - If there are conflicts with keyword names, these keywords have\n          *lower* precedence than other keywords.\n\n        This API is pretty bad and will be reimplemented in the future.\n        It is thus not considered stable, but external libraries can use it\n        if they really need it and are aware of forthcoming breaking changes.\n\n        Something like this is needed at least internally also in the future.\n        For external libraries we hopefully could provide a better API for\n        running keywords so that they would not need this in the first place.\n\n        For more details see the following issues and issues linked from it:\n        https://github.com/robotframework/robotframework/issues/2190\n\n        :param libname: Name of the library the keyword belongs to.\n        :param keyword: Name of the keyword itself.\n        :param args_to_process: How many arguments to process normally before\n            passing them to the keyword. Other arguments are not touched at all.\n        :param dry_run: When true, this keyword is executed in dry run. Keywords\n            to actually run are got based on the ``name`` argument these\n            keywords must have.\n        :param deprecation_warning: Set to ``False```to avoid the warning.\n        '
        if deprecation_warning:
            warnings.warn('The API to register run keyword variants and to disable variable resolving in keyword arguments will change in the future. For more information see https://github.com/robotframework/robotframework/issues/2190. Use with `deprecation_warning=False` to avoid this warning.', UserWarning)
        if libname not in self._libs:
            self._libs[libname] = NormalizedDict(ignore=['_'])
        self._libs[libname][keyword] = (int(args_to_process), dry_run)

    def get_args_to_process(self, libname, kwname):
        if False:
            print('Hello World!')
        if libname in self._libs and kwname in self._libs[libname]:
            return self._libs[libname][kwname][0]
        return -1

    def get_dry_run(self, libname, kwname):
        if False:
            return 10
        if libname in self._libs and kwname in self._libs[libname]:
            return self._libs[libname][kwname][1]
        return False

    def is_run_keyword(self, libname, kwname):
        if False:
            while True:
                i = 10
        return self.get_args_to_process(libname, kwname) >= 0
RUN_KW_REGISTER = _RunKeywordRegister()