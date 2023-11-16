from coalib.core.Bear import Bear
from coalib.settings.FunctionMetadata import FunctionMetadata

class DependencyBear(Bear):
    """
    This bear base class parallelizes tasks for each dependency result.

    You can specify dependency bears with the ``BEAR_DEPS`` field.
    """

    def __init__(self, section, file_dict):
        if False:
            i = 10
            return i + 15
        '\n        :param section:\n            The section object where bear settings are contained. A section\n            passed here is considered to be immutable.\n        :param file_dict:\n            A dictionary containing filenames to process as keys and their\n            contents (line-split with trailing return characters) as values.\n        '
        Bear.__init__(self, section, file_dict)
        self._kwargs = self.get_metadata().create_params_from_section(section)

    @classmethod
    def get_metadata(cls):
        if False:
            while True:
                i = 10
        '\n        :return:\n            Metadata for the ``analyze`` function extracted from its signature.\n            Excludes parameters ``self``, ``filename`` and ``file``.\n        '
        return FunctionMetadata.from_function(cls.analyze, omit={'self', 'dependency_bear', 'dependency_result'})

    def generate_tasks(self):
        if False:
            for i in range(10):
                print('nop')
        return (((bear, dependency_result), self._kwargs) for (bear, dependency_results) in self.dependency_results.items() for dependency_result in dependency_results)