from typing import List, Dict
from pydantic import ValidationError
from pydantic import BaseModel
from pandasai.helpers.df_info import DataFrameType, df_type

class DfValidationResult:
    """
    Validation results for a dataframe.

    Attributes:
        passed: Whether the validation passed or not.
        errors: List of errors if the validation failed.
    """
    _passed: bool
    _errors: List[Dict]

    def __init__(self, passed: bool=True, errors: List[Dict]=None):
        if False:
            while True:
                i = 10
        '\n        Args:\n            passed: Whether the validation passed or not.\n            errors: List of errors if the validation failed.\n        '
        if errors is None:
            errors = []
        self._passed = passed
        self._errors = errors

    @property
    def passed(self):
        if False:
            for i in range(10):
                print('nop')
        return self._passed

    def errors(self) -> List[Dict]:
        if False:
            print('Hello World!')
        return self._errors

    def add_error(self, error_message: str):
        if False:
            i = 10
            return i + 15
        '\n        Add an error message to the validation results.\n\n        Args:\n            error_message: Error message to add.\n        '
        self._passed = False
        self._errors.append(error_message)

    def __bool__(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Define the truthiness of ValidationResults.\n        '
        return self.passed

class DfValidator:
    """
    Validate a dataframe using a Pydantic schema.

    Attributes:
        df: dataframe to be validated
    """
    _df: DataFrameType

    def __init__(self, df: DataFrameType):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            df: dataframe to be validated\n        '
        self._df = df

    def _validate_batch(self, schema, df_json: List[Dict]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            schema: Pydantic schema\n            batch_df: dataframe batch\n\n        Returns:\n            list of errors\n        '
        try:

            class PdVal(BaseModel):
                df: List[schema]
            PdVal(df=df_json)
            return []
        except ValidationError as e:
            return e.errors()

    def _df_to_list_of_dict(self, df: DataFrameType, dataframe_type: str) -> List[Dict]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create list of dict of dataframe rows on basis of dataframe type\n        Supports only polars and pandas dataframe\n\n        Args:\n            df: dataframe to be converted\n            dataframe_type: type of dataframe\n\n        Returns:\n            list of dict of dataframe rows\n        '
        if dataframe_type == 'pandas':
            return df.to_dict(orient='records')
        elif dataframe_type == 'polars':
            return df.to_dicts()
        else:
            return []

    def validate(self, schema: BaseModel) -> DfValidationResult:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            schema: Pydantic schema to be validated for the dataframe row\n\n        Returns:\n            Validation results\n        '
        dataframe_type = df_type(self._df)
        if dataframe_type is None:
            raise ValueError('Unsupported DataFrame')
        df_json: List[Dict] = self._df_to_list_of_dict(self._df, dataframe_type)
        errors = self._validate_batch(schema, df_json)
        return DfValidationResult(len(errors) == 0, errors)