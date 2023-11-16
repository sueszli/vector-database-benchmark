"""TemplateEntity utility class."""
from __future__ import annotations
import contextlib
import logging
from typing import Any
import voluptuous as vol
from homeassistant.components.sensor import CONF_STATE_CLASS, DEVICE_CLASSES_SCHEMA, STATE_CLASSES_SCHEMA, SensorEntity
from homeassistant.const import ATTR_ENTITY_PICTURE, ATTR_FRIENDLY_NAME, ATTR_ICON, CONF_DEVICE_CLASS, CONF_ICON, CONF_NAME, CONF_UNIQUE_ID, CONF_UNIT_OF_MEASUREMENT
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.exceptions import TemplateError
from homeassistant.util.json import JSON_DECODE_EXCEPTIONS, json_loads
from . import config_validation as cv
from .entity import Entity
from .template import attach as template_attach, render_complex
from .typing import ConfigType
CONF_AVAILABILITY = 'availability'
CONF_ATTRIBUTES = 'attributes'
CONF_PICTURE = 'picture'
CONF_TO_ATTRIBUTE = {CONF_ICON: ATTR_ICON, CONF_NAME: ATTR_FRIENDLY_NAME, CONF_PICTURE: ATTR_ENTITY_PICTURE}
TEMPLATE_ENTITY_BASE_SCHEMA = vol.Schema({vol.Optional(CONF_ICON): cv.template, vol.Optional(CONF_NAME): cv.template, vol.Optional(CONF_PICTURE): cv.template, vol.Optional(CONF_UNIQUE_ID): cv.string})

def make_template_entity_base_schema(default_name: str) -> vol.Schema:
    if False:
        print('Hello World!')
    'Return a schema with default name.'
    return vol.Schema({vol.Optional(CONF_ICON): cv.template, vol.Optional(CONF_NAME, default=default_name): cv.template, vol.Optional(CONF_PICTURE): cv.template, vol.Optional(CONF_UNIQUE_ID): cv.string})
TEMPLATE_SENSOR_BASE_SCHEMA = vol.Schema({vol.Optional(CONF_DEVICE_CLASS): DEVICE_CLASSES_SCHEMA, vol.Optional(CONF_STATE_CLASS): STATE_CLASSES_SCHEMA, vol.Optional(CONF_UNIT_OF_MEASUREMENT): cv.string}).extend(TEMPLATE_ENTITY_BASE_SCHEMA.schema)

class TriggerBaseEntity(Entity):
    """Template Base entity based on trigger data."""
    domain: str
    extra_template_keys: tuple[str, ...] | None = None
    extra_template_keys_complex: tuple[str, ...] | None = None
    _unique_id: str | None

    def __init__(self, hass: HomeAssistant, config: ConfigType) -> None:
        if False:
            return 10
        'Initialize the entity.'
        self.hass = hass
        self._set_unique_id(config.get(CONF_UNIQUE_ID))
        self._config = config
        self._static_rendered = {}
        self._to_render_simple: list[str] = []
        self._to_render_complex: list[str] = []
        for itm in (CONF_AVAILABILITY, CONF_ICON, CONF_NAME, CONF_PICTURE):
            if itm not in config or config[itm] is None:
                continue
            if config[itm].is_static:
                self._static_rendered[itm] = config[itm].template
            else:
                self._to_render_simple.append(itm)
        if self.extra_template_keys is not None:
            self._to_render_simple.extend(self.extra_template_keys)
        if self.extra_template_keys_complex is not None:
            self._to_render_complex.extend(self.extra_template_keys_complex)
        self._rendered = dict(self._static_rendered)
        self._parse_result = {CONF_AVAILABILITY}
        self._attr_device_class = config.get(CONF_DEVICE_CLASS)

    @property
    def name(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        'Name of the entity.'
        return self._rendered.get(CONF_NAME)

    @property
    def unique_id(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        'Return unique ID of the entity.'
        return self._unique_id

    @property
    def icon(self) -> str | None:
        if False:
            while True:
                i = 10
        'Return icon.'
        return self._rendered.get(CONF_ICON)

    @property
    def entity_picture(self) -> str | None:
        if False:
            while True:
                i = 10
        'Return entity picture.'
        return self._rendered.get(CONF_PICTURE)

    @property
    def available(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Return availability of the entity.'
        return self._rendered is not self._static_rendered and self._rendered.get(CONF_AVAILABILITY) is not False

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        if False:
            i = 10
            return i + 15
        'Return extra attributes.'
        return self._rendered.get(CONF_ATTRIBUTES)

    async def async_added_to_hass(self) -> None:
        """Handle being added to Home Assistant."""
        await super().async_added_to_hass()
        template_attach(self.hass, self._config)

    def _set_unique_id(self, unique_id: str | None) -> None:
        if False:
            i = 10
            return i + 15
        'Set unique id.'
        self._unique_id = unique_id

    def restore_attributes(self, last_state: State) -> None:
        if False:
            i = 10
            return i + 15
        'Restore attributes.'
        for (conf_key, attr) in CONF_TO_ATTRIBUTE.items():
            if conf_key not in self._config or attr not in last_state.attributes:
                continue
            self._rendered[conf_key] = last_state.attributes[attr]
        if CONF_ATTRIBUTES in self._config:
            extra_state_attributes = {}
            for attr in self._config[CONF_ATTRIBUTES]:
                if attr not in last_state.attributes:
                    continue
                extra_state_attributes[attr] = last_state.attributes[attr]
            self._rendered[CONF_ATTRIBUTES] = extra_state_attributes

    def _render_templates(self, variables: dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Render templates.'
        try:
            rendered = dict(self._static_rendered)
            for key in self._to_render_simple:
                rendered[key] = self._config[key].async_render(variables, parse_result=key in self._parse_result)
            for key in self._to_render_complex:
                rendered[key] = render_complex(self._config[key], variables)
            if CONF_ATTRIBUTES in self._config:
                rendered[CONF_ATTRIBUTES] = render_complex(self._config[CONF_ATTRIBUTES], variables)
            self._rendered = rendered
        except TemplateError as err:
            logging.getLogger(f"{__package__}.{self.entity_id.split('.')[0]}").error('Error rendering %s template for %s: %s', key, self.entity_id, err)
            self._rendered = self._static_rendered

class ManualTriggerEntity(TriggerBaseEntity):
    """Template entity based on manual trigger data."""

    def __init__(self, hass: HomeAssistant, config: ConfigType) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize the entity.'
        TriggerBaseEntity.__init__(self, hass, config)
        self._rendered[CONF_NAME] = config[CONF_NAME].async_render({}, parse_result=CONF_NAME in self._parse_result)

    @callback
    def _process_manual_data(self, value: Any | None=None) -> None:
        if False:
            i = 10
            return i + 15
        'Process new data manually.\n\n        Implementing class should call this last in update method to render templates.\n        Ex: self._process_manual_data(payload)\n        '
        self.async_write_ha_state()
        this = None
        if (state := self.hass.states.get(self.entity_id)):
            this = state.as_dict()
        run_variables: dict[str, Any] = {'value': value}
        with contextlib.suppress(*JSON_DECODE_EXCEPTIONS):
            run_variables['value_json'] = json_loads(run_variables['value'])
        variables = {'this': this, **(run_variables or {})}
        self._render_templates(variables)

class ManualTriggerSensorEntity(ManualTriggerEntity, SensorEntity):
    """Template entity based on manual trigger data for sensor."""

    def __init__(self, hass: HomeAssistant, config: ConfigType) -> None:
        if False:
            print('Hello World!')
        'Initialize the sensor entity.'
        ManualTriggerEntity.__init__(self, hass, config)
        self._attr_native_unit_of_measurement = config.get(CONF_UNIT_OF_MEASUREMENT)
        self._attr_state_class = config.get(CONF_STATE_CLASS)