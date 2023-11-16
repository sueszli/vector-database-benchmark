from abc import ABC, abstractmethod
from typing import Any
from pandasai.helpers.env import is_running_in_console
from ..helpers.df_info import polars_imported
from pandasai.exceptions import MethodNotImplementedError

class IResponseParser(ABC):

    @abstractmethod
    def parse(self, result: dict) -> Any:
        if False:
            while True:
                i = 10
        '\n        Parses result from the chat input\n        Args:\n            result (dict): result contains type and value\n        Raises:\n            ValueError: if result is not a dictionary with valid key\n\n        Returns:\n            Any: Returns depending on the user input\n        '
        raise MethodNotImplementedError

class ResponseParser(IResponseParser):
    _context = None

    def __init__(self, context) -> None:
        if False:
            print('Hello World!')
        '\n        Initialize the ResponseParser with Context from SmartDataLake\n        Args:\n            context (Context): context contains the config, logger and engine\n        '
        self._context = context

    def parse(self, result: dict) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses result from the chat input\n        Args:\n            result (dict): result contains type and value\n        Raises:\n            ValueError: if result is not a dictionary with valid key\n\n        Returns:\n            Any: Returns depending on the user input\n        '
        if not isinstance(result, dict) or any((key not in result for key in ['type', 'value'])):
            raise ValueError('Unsupported result format')
        if result['type'] == 'dataframe':
            return self.format_dataframe(result)
        elif result['type'] == 'plot':
            return self.format_plot(result)
        else:
            return self.format_other(result)

    def format_dataframe(self, result: dict) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Format dataframe generate against a user query\n        Args:\n            result (dict): result contains type and value\n        Returns:\n            Any: Returns depending on the user input\n        '
        from ..smart_dataframe import SmartDataframe
        df = result['value']
        if self._context.engine == 'polars' and polars_imported:
            import polars as pl
            df = pl.from_pandas(df)
        return SmartDataframe(df, config=self._context._config.__dict__, logger=self._context.logger)

    def format_plot(self, result: dict) -> Any:
        if False:
            return 10
        '\n        Display matplotlib plot against a user query\n        Args:\n            result (dict): result contains type and value\n        Returns:\n            Any: Returns depending on the user input\n        '
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        try:
            image = mpimg.imread(result['value'])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The file {result['value']} does not exist.") from e
        except OSError as e:
            raise ValueError(f"The file {result['value']} is not a valid image file.") from e
        plt.imshow(image)
        plt.axis('off')
        plt.show(block=is_running_in_console())
        plt.close('all')

    def format_other(self, result) -> Any:
        if False:
            print('Hello World!')
        '\n        Returns the result generated against a user query other than dataframes\n        and plots\n        Args:\n            result (dict): result contains type and value\n        Returns:\n            Any: Returns depending on the user input\n        '
        return result['value']