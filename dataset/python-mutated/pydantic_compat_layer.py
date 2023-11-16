from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import pydantic
from pydantic import BaseModel
from .attach_other_object_to_context import IAttachDifferentObjectToOpContext as IAttachDifferentObjectToOpContext
USING_PYDANTIC_2 = int(pydantic.__version__.split('.')[0]) >= 2
PydanticUndefined = None
if USING_PYDANTIC_2:
    from pydantic_core import PydanticUndefined as _PydanticUndefined
    PydanticUndefined = _PydanticUndefined
if TYPE_CHECKING:
    from pydantic.fields import ModelField

class ModelFieldCompat:
    """Wraps a Pydantic model field to provide a consistent interface for accessing
    metadata and annotations between Pydantic 1 and 2.
    """

    def __init__(self, field) -> None:
        if False:
            return 10
        self.field: 'ModelField' = field

    @property
    def annotation(self) -> Type:
        if False:
            while True:
                i = 10
        return self.field.annotation

    @property
    def metadata(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        return getattr(self.field, 'metadata', [])

    @property
    def alias(self) -> str:
        if False:
            print('Hello World!')
        return self.field.alias

    @property
    def default(self) -> Any:
        if False:
            while True:
                i = 10
        return self.field.default

    @property
    def description(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        if USING_PYDANTIC_2:
            return getattr(self.field, 'description', None)
        else:
            field_info = getattr(self.field, 'field_info', None)
            return field_info.description if field_info else None

    def is_required(self) -> bool:
        if False:
            i = 10
            return i + 15
        if USING_PYDANTIC_2:
            return self.field.is_required()
        else:
            return self.field.required if isinstance(self.field.required, bool) else False

    @property
    def discriminator(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        if USING_PYDANTIC_2:
            if hasattr(self.field, 'discriminator'):
                return self.field.discriminator if hasattr(self.field, 'discriminator') else None
        else:
            return getattr(self.field, 'discriminator_key', None)

def model_fields(model) -> Dict[str, ModelFieldCompat]:
    if False:
        i = 10
        return i + 15
    'Returns a dictionary of fields for a given pydantic model, wrapped\n    in a compat class to provide a consistent interface between Pydantic 1 and 2.\n    '
    fields = getattr(model, 'model_fields', None)
    if not fields:
        fields = getattr(model, '__fields__')
    return {k: ModelFieldCompat(v) for (k, v) in fields.items()}

class Pydantic1ConfigWrapper:
    """Config wrapper for Pydantic 1 style model config, which provides a
    Pydantic 2 style interface for accessing mopdel config values.
    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        self._config = config

    def get(self, key):
        if False:
            while True:
                i = 10
        return getattr(self._config, key)

def model_config(model: Type[BaseModel]):
    if False:
        for i in range(10):
            print('nop')
    'Returns the config for a given pydantic model, wrapped such that it has\n    a Pydantic 2-style interface for accessing config values.\n    '
    if USING_PYDANTIC_2:
        return getattr(model, 'model_config')
    else:
        return Pydantic1ConfigWrapper(getattr(model, '__config__'))
try:
    from pydantic import model_validator as model_validator
except ImportError:
    from pydantic import root_validator

    def model_validator(mode='before'):
        if False:
            print('Hello World!')
        'Mimics the Pydantic 2.x model_validator decorator, which is used to\n        define validation logic for a Pydantic model. This decorator is used\n        to wrap a validation function which is called before or after the\n        model is constructed.\n        '

        def _decorate(func):
            if False:
                print('Hello World!')
            return root_validator(pre=True)(func) if mode == 'before' else root_validator(post=False)(func)
        return _decorate
compat_model_validator = model_validator