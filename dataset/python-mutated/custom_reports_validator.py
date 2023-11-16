from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, ValidationError, validator
from pydantic.class_validators import root_validator
ERROR_MSG_MISSING_SEGMENT_DIMENSION = 'errors: `ga:segment` is required'

class Model(BaseModel):

    class Config:
        extra = 'forbid'
    name: str
    dimensions: list[str]
    metrics: list[str]
    filter: Optional[str]
    segments: Optional[list[str]]

    @validator('dimensions', 'metrics')
    def check_field_reference_forrmat(cls, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Defines rules for nested strings, for fields: dimensions, metrics.\n        General rule: the `ga:` prefix is defined for each field\n        '
        for v in value:
            if 'ga:' not in v:
                raise ValueError(v)

    @classmethod
    @root_validator(pre=True)
    def check_segment_included_in_dimension(cls, values):
        if False:
            for i in range(10):
                print('nop')
        dimensions = values.get('dimensions')
        segments = values.get('segments')
        if segments and 'ga:segment' not in dimensions:
            raise ValueError(ERROR_MSG_MISSING_SEGMENT_DIMENSION)
        return values

class Explainer:
    """
    ERRORS_MAPPING holds an external `Pydantic.ValidationError` types and their placeholders.
    {
        key: str = <Pydantic.ValidationError Type>,
        value: tuple(str, list) = (<explainable message>, <list as placeholder>
    }

    """
    errors_mapping = {'value_error.missing': ('fields required', []), 'value_error.extra': ('fields not permitted', []), 'type_error': ('type errors', []), 'value_error': ('incorrect field reference, expected format `ga:MY_FIELD_NAME`, but got', [])}

    def parse(self, errors: List[Dict]) -> str:
        if False:
            print('Hello World!')
        for error in errors:
            (field_name, error_type, error_msg) = (error.get('loc')[0], error.get('type'), error.get('msg'))
            if error_type in self.errors_mapping:
                if error_type == 'value_error':
                    self.errors_mapping.get(error_type)[1].append({'field': field_name, 'reference': error_msg})
                else:
                    self.errors_mapping.get(error_type)[1].append(field_name)
            if 'type_error' in error_type:
                (error_type, _type) = error_type.split('.')
                self.errors_mapping.get(error_type)[1].append((field_name, f'{_type} is required'))

    def explain(self, errors: List[Dict]):
        if False:
            while True:
                i = 10
        '\n        General Errors are explained first.\n        Such as:\n            - missing required field\n            - presence of non-permitted fields\n\n        Type Errors are explained last.\n        If model attribute has invalid type provided, like list, but str was required and etc:\n            - str is required,\n            - ...\n        '
        self.parse(errors)
        for error_type in self.errors_mapping:
            (msg, errors) = self.errors_mapping.get(error_type)
            if errors:
                return f'{msg} {errors}'

@dataclass
class CustomReportsValidator:
    custom_reports: Union[List[Dict], Dict] = Field(default_factory=list)

    def __post_init__(self):
        if False:
            while True:
                i = 10
        self.reports: list = [self.custom_reports] if not isinstance(self.custom_reports, list) else self.custom_reports
        self.model: Model = Model
        self.explainer: Explainer = Explainer()

    def validate(self):
        if False:
            for i in range(10):
                print('nop')
        from airbyte_cdk.models import FailureType
        from airbyte_cdk.utils.traced_exception import AirbyteTracedException
        try:
            for report in self.reports:
                self.model.parse_obj(report)
        except ValidationError as e:
            raise AirbyteTracedException(message=None, internal_message=f'Custom Reports has invalid structure in report: {report}, errors: {self.explainer.explain(e.errors())}', failure_type=FailureType.config_error) from None