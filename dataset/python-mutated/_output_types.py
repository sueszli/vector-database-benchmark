import re
from decimal import Decimal
from abc import abstractmethod, ABC
from typing import Any, Iterable
from ..df_info import df_type

class BaseOutputType(ABC):

    @property
    @abstractmethod
    def template_hint(self) -> str:
        if False:
            i = 10
            return i + 15
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        ...

    def _validate_type(self, actual_type: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return actual_type == self.name

    @abstractmethod
    def _validate_value(self, actual_value):
        if False:
            print('Hello World!')
        ...

    def validate(self, result: dict[str, Any]) -> tuple[bool, Iterable[str]]:
        if False:
            while True:
                i = 10
        '\n        Validate \'type\' and \'value\' from the result dict.\n\n        Args:\n            result (dict[str, Any]): The result of code execution in\n                dict representation. Should have the following schema:\n                {\n                    "type": <output_type_name>,\n                    "value": <generated_value>\n                }\n\n        Returns:\n             (tuple(bool, Iterable(str)):\n                Boolean value whether the result matches output type\n                and collection of logs containing messages about\n                \'type\' or \'value\' mismatches.\n        '
        validation_logs = []
        (actual_type, actual_value) = (result.get('type'), result.get('value'))
        type_ok = self._validate_type(actual_type)
        if not type_ok:
            validation_logs.append(f"The result dict contains inappropriate 'type'. Expected '{self.name}', actual '{actual_type}'.")
        value_ok = self._validate_value(actual_value)
        if not value_ok:
            validation_logs.append(f"Actual value {repr(actual_value)} seems to be inappropriate for the type '{self.name}'.")
        return (all((type_ok, value_ok)), validation_logs)

class NumberOutputType(BaseOutputType):

    @property
    def template_hint(self):
        if False:
            while True:
                i = 10
        return '- type (must be "number")\n    - value (must be a number)\n    Example output: { "type": "number", "value": 125 }'

    @property
    def name(self):
        if False:
            print('Hello World!')
        return 'number'

    def _validate_value(self, actual_value: Any) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(actual_value, (int, float, Decimal))

class DataFrameOutputType(BaseOutputType):

    @property
    def template_hint(self):
        if False:
            for i in range(10):
                print('nop')
        return '- type (must be "dataframe")\n    - value (must be a pandas dataframe)\n    Example output: { "type": "dataframe", "value": pd.DataFrame({...}) }'

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return 'dataframe'

    def _validate_value(self, actual_value: Any) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(df_type(actual_value))

class PlotOutputType(BaseOutputType):

    @property
    def template_hint(self):
        if False:
            i = 10
            return i + 15
        return '- type (must be "plot")\n    - value (must be a string containing the path of the plot image)\n    Example output: { "type": "plot", "value": "export/charts/temp_chart.png" }'

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return 'plot'

    def _validate_value(self, actual_value: Any) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(actual_value, str):
            return False
        path_to_plot_pattern = '^(\\/[\\w.-]+)+(/[\\w.-]+)*$|^[^\\s/]+(/[\\w.-]+)*$'
        return bool(re.match(path_to_plot_pattern, actual_value))

class StringOutputType(BaseOutputType):

    @property
    def template_hint(self):
        if False:
            for i in range(10):
                print('nop')
        return '- type (must be "string")\n    - value (must be a conversational answer, as a string)\n    Example output: { "type": "string", "value": f"The highest salary is {highest_salary}." }'

    @property
    def name(self):
        if False:
            print('Hello World!')
        return 'string'

    def _validate_value(self, actual_value: Any) -> bool:
        if False:
            return 10
        return isinstance(actual_value, str)

class DefaultOutputType(BaseOutputType):

    @property
    def template_hint(self):
        if False:
            print('Hello World!')
        return '- type (possible values "string", "number", "dataframe", "plot")\n    - value (can be a string, a dataframe or the path of the plot, NOT a dictionary)\n    Examples: \n        { "type": "string", "value": f"The highest salary is {highest_salary}." }\n        or\n        { "type": "number", "value": 125 }\n        or\n        { "type": "dataframe", "value": pd.DataFrame({...}) }\n        or\n        { "type": "plot", "value": "temp_chart.png" }'

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return 'default'

    def _validate_type(self, actual_type: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def _validate_value(self, actual_value: Any) -> bool:
        if False:
            while True:
                i = 10
        return True

    def validate(self, result: dict[str, Any]) -> tuple[bool, Iterable]:
        if False:
            i = 10
            return i + 15
        "\n        Validate 'type' and 'value' from the result dict.\n\n        Returns:\n             (bool): True since the `DefaultOutputType`\n                is supposed to have no validation\n        "
        return (True, ())