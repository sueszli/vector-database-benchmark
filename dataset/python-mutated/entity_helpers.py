import esphome.final_validate as fv
from esphome.const import CONF_ID

def inherit_property_from(property_to_inherit, parent_id_property, transform=None):
    if False:
        return 10
    'Validator that inherits a configuration property from another entity, for use with FINAL_VALIDATE_SCHEMA.\n    If a property is already set, it will not be inherited.\n    Keyword arguments:\n    property_to_inherit -- the name or path of the property to inherit, e.g. CONF_ICON or [CONF_SENSOR, 0, CONF_ICON]\n                           (the parent must exist, otherwise nothing is done).\n    parent_id_property -- the name or path of the property that holds the ID of the parent, e.g. CONF_POWER_ID or\n                          [CONF_SENSOR, 1, CONF_POWER_ID].\n    '

    def _walk_config(config, path):
        if False:
            return 10
        walk = [path] if not isinstance(path, list) else path
        for item_or_index in walk:
            config = config[item_or_index]
        return config

    def inherit_property(config):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(property_to_inherit, list):
            (property_path, property) = ([], property_to_inherit)
        else:
            (property_path, property) = (property_to_inherit[:-1], property_to_inherit[-1])
        try:
            config_part = _walk_config(config, property_path)
        except KeyError:
            return config
        if property not in config_part:
            fconf = fv.full_config.get()
            parent_id = _walk_config(config, parent_id_property)
            parent_path = fconf.get_path_for_id(parent_id)[:-1]
            parent_config = fconf.get_config_for_path(parent_path)
            if property in parent_config:
                path = fconf.get_path_for_id(config[CONF_ID])[:-1]
                this_config = _walk_config(fconf.get_config_for_path(path), property_path)
                value = parent_config[property]
                if transform:
                    value = transform(value, config)
                this_config[property] = value
        return config
    return inherit_property