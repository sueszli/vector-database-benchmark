import re

class ProcessErrorExtractor:

    def __init__(self, error: str):
        if False:
            i = 10
            return i + 15
        "\n        expecting an error like this:\n            ModuleNotFoundError: No module named 'mymodule'\n        "
        self.error = error

    def is_import_error(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return 'ModuleNotFoundError' in self.error

    def get_missing_package_name(self) -> str:
        if False:
            return 10
        '\n        Returns a name of a module that extension failed to import\n        '
        match = re.match("^.*'(\\w+)['\\.]", self.error)
        return match.group(1) if match else ''