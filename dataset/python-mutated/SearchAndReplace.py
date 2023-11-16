import re
from ..Script import Script

class SearchAndReplace(Script):
    """Performs a search-and-replace on all g-code.

    Due to technical limitations, the search can't cross the border between
    layers.
    """

    def getSettingDataString(self):
        if False:
            i = 10
            return i + 15
        return '{\n            "name": "Search and Replace",\n            "key": "SearchAndReplace",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "search":\n                {\n                    "label": "Search",\n                    "description": "All occurrences of this text will get replaced by the replacement text.",\n                    "type": "str",\n                    "default_value": ""\n                },\n                "replace":\n                {\n                    "label": "Replace",\n                    "description": "The search text will get replaced by this text.",\n                    "type": "str",\n                    "default_value": ""\n                },\n                "is_regex":\n                {\n                    "label": "Use Regular Expressions",\n                    "description": "When enabled, the search text will be interpreted as a regular expression.",\n                    "type": "bool",\n                    "default_value": false\n                }\n            }\n        }'

    def execute(self, data):
        if False:
            return 10
        search_string = self.getSettingValueByKey('search')
        if not self.getSettingValueByKey('is_regex'):
            search_string = re.escape(search_string)
        search_regex = re.compile(search_string)
        replace_string = self.getSettingValueByKey('replace')
        for (layer_number, layer) in enumerate(data):
            data[layer_number] = re.sub(search_regex, replace_string, layer)
        return data