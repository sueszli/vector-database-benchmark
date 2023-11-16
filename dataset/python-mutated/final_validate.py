from abc import ABC, abstractmethod
from typing import Any
import contextvars
from esphome.types import ConfigFragmentType, ID, ConfigPathType
import esphome.config_validation as cv

class FinalValidateConfig(ABC):

    @property
    @abstractmethod
    def data(self) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        'A dictionary that can be used by post validation functions to store\n        global data during the validation phase. Each component should store its\n        data under a unique key\n        '

    @abstractmethod
    def get_path_for_id(self, id: ID) -> ConfigPathType:
        if False:
            return 10
        'Get the config path a given ID has been declared in.\n\n        This is the location under the _validated_ config (for example, with cv.ensure_list applied)\n        Raises KeyError if the id was not declared in the configuration.\n        '

    @abstractmethod
    def get_config_for_path(self, path: ConfigPathType) -> ConfigFragmentType:
        if False:
            i = 10
            return i + 15
        'Get the config fragment for the given global path.\n\n        Raises KeyError if a key in the path does not exist.\n        '
FinalValidateConfig.register(dict)
full_config: contextvars.ContextVar[FinalValidateConfig] = contextvars.ContextVar('full_config')

def id_declaration_match_schema(schema):
    if False:
        for i in range(10):
            print('nop')
    'A final-validation schema function that applies a schema to the outer config fragment of an\n    ID declaration.\n\n    This validator must be applied to ID values.\n    '
    if not isinstance(schema, cv.Schema):
        schema = cv.Schema(schema, extra=cv.ALLOW_EXTRA)

    def validator(value):
        if False:
            i = 10
            return i + 15
        fconf = full_config.get()
        path = fconf.get_path_for_id(value)[:-1]
        declaration_config = fconf.get_config_for_path(path)
        with cv.prepend_path([cv.ROOT_CONFIG_PATH] + path):
            return schema(declaration_config)
    return validator