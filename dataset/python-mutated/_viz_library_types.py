from abc import abstractmethod, ABC
from typing import Any, Iterable
from pandasai.prompts.generate_python_code import VizLibraryPrompt

class BaseVizLibraryType(ABC):

    @property
    def template_hint(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return VizLibraryPrompt(library=self.name)

    @property
    @abstractmethod
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        ...

    def _validate_type(self, actual_type: str) -> bool:
        if False:
            while True:
                i = 10
        return actual_type == self.name

    def validate(self, result: dict[str, Any]) -> tuple[bool, Iterable[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Validate \'type\' and \'constraint\' from the result dict.\n\n        Args:\n            result (dict[str, Any]): The result of code execution in\n                dict representation. Should have the following schema:\n                {\n                    "viz_library_type": <viz_library_name>\n                }\n\n        Returns:\n             (tuple(bool, Iterable(str)):\n                Boolean value whether the result matches output type\n                and collection of logs containing messages about\n                \'type\' mismatches.\n        '
        validation_logs = []
        actual_type = result.get('type')
        type_ok = self._validate_type(actual_type)
        if not type_ok:
            validation_logs.append(f"The result dict contains inappropriate 'type'. Expected '{self.name}', actual '{actual_type}'.")
        return (type_ok, validation_logs)

class MatplotlibVizLibraryType(BaseVizLibraryType):

    @property
    def name(self):
        if False:
            return 10
        return 'matplotlib'

class PlotlyVizLibraryType(BaseVizLibraryType):

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return 'plotly'

class SeabornVizLibraryType(BaseVizLibraryType):

    @property
    def name(self):
        if False:
            return 10
        return 'seaborn'