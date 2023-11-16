"""Helpers for Z-Wave JS custom triggers."""
from zwave_js_server.client import Client as ZwaveClient
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import ATTR_DEVICE_ID, ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.typing import ConfigType
from ..const import ATTR_CONFIG_ENTRY_ID, DATA_CLIENT, DOMAIN

@callback
def async_bypass_dynamic_config_validation(hass: HomeAssistant, config: ConfigType) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Return whether target zwave_js config entry is not loaded.'
    dev_reg = dr.async_get(hass)
    ent_reg = er.async_get(hass)
    trigger_devices = config.get(ATTR_DEVICE_ID, [])
    trigger_entities = config.get(ATTR_ENTITY_ID, [])
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.state != ConfigEntryState.LOADED and (entry.entry_id == config.get(ATTR_CONFIG_ENTRY_ID) or any((device.id in trigger_devices for device in dr.async_entries_for_config_entry(dev_reg, entry.entry_id))) or (entity.entity_id in trigger_entities for entity in er.async_entries_for_config_entry(ent_reg, entry.entry_id))):
            return True
        client: ZwaveClient = hass.data[DOMAIN][entry.entry_id][DATA_CLIENT]
        if client.driver is None:
            return True
    return False