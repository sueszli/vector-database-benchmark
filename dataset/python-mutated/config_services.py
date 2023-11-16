"""Services for configuration properties."""
from __future__ import annotations
from core.domain import config_domain
from core.platform import models
from typing import Any
(config_models,) = models.Registry.import_models([models.Names.CONFIG])
CMD_CHANGE_PROPERTY_VALUE = 'change_property_value'

def set_property(committer_id: str, name: str, value: Any) -> None:
    if False:
        print('Hello World!')
    'Sets a property value. The property must already be registered.\n\n    Args:\n        committer_id: str. The user ID of the committer.\n        name: str. The name of the property.\n        value: Any. The value of the property.\n\n    Raises:\n        Exception. No config property with the specified name is found.\n    '
    config_property = config_domain.Registry.get_config_property(name)
    if config_property is None:
        raise Exception('No config property with name %s found.' % name)
    config_property.set_value(committer_id, value)

def revert_property(committer_id: str, name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Reverts a property value to the default value.\n\n    Args:\n        committer_id: str. The user ID of the committer.\n        name: str. The name of the property.\n\n    Raises:\n        Exception. No config property with the specified name is found.\n    '
    config_property = config_domain.Registry.get_config_property(name)
    if config_property is None:
        raise Exception('No config property with name %s found.' % name)
    set_property(committer_id, name, config_property.default_value)