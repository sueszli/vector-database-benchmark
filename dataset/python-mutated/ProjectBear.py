from coalib.core.Bear import Bear
from coalib.settings.FunctionMetadata import FunctionMetadata

class ProjectBear(Bear):
    """
    This bear base class does not parallelize tasks at all, it runs on the
    whole file base provided.
    """

    def __init__(self, section, file_dict):
        if False:
            print('Hello World!')
        '\n        :param section:\n            The section object where bear settings are contained. A section\n            passed here is considered to be immutable.\n        :param file_dict:\n            A dictionary containing filenames to process as keys and their\n            contents (line-split with trailing return characters) as values.\n        '
        Bear.__init__(self, section, file_dict)
        self._kwargs = self.get_metadata().create_params_from_section(section)

    @classmethod
    def get_metadata(cls):
        if False:
            i = 10
            return i + 15
        '\n        :return:\n            Metadata for the ``analyze`` function extracted from its signature.\n            Excludes parameters ``self`` and ``files``.\n        '
        return FunctionMetadata.from_function(cls.analyze, omit={'self', 'files'})

    def generate_tasks(self):
        if False:
            for i in range(10):
                print('nop')
        return (((self.file_dict,), self._kwargs),)