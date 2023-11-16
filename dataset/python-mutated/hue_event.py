"""Handle forward of events transmitted by Hue devices to HASS."""
import logging
from typing import TYPE_CHECKING
from aiohue.v2 import HueBridgeV2
from aiohue.v2.controllers.events import EventType
from aiohue.v2.models.button import Button
from aiohue.v2.models.relative_rotary import RelativeRotary
from homeassistant.const import CONF_DEVICE_ID, CONF_ID, CONF_TYPE, CONF_UNIQUE_ID
from homeassistant.core import callback
from homeassistant.helpers import device_registry as dr
from homeassistant.util import slugify
from ..const import ATTR_HUE_EVENT, CONF_SUBTYPE, DOMAIN
CONF_CONTROL_ID = 'control_id'
CONF_DURATION = 'duration'
CONF_STEPS = 'steps'
if TYPE_CHECKING:
    from ..bridge import HueBridge
LOGGER = logging.getLogger(__name__)

async def async_setup_hue_events(bridge: 'HueBridge'):
    """Manage listeners for stateless Hue sensors that emit events."""
    hass = bridge.hass
    api: HueBridgeV2 = bridge.api
    conf_entry = bridge.config_entry
    dev_reg = dr.async_get(hass)
    btn_controller = api.sensors.button
    rotary_controller = api.sensors.relative_rotary

    @callback
    def handle_button_event(evt_type: EventType, hue_resource: Button) -> None:
        if False:
            return 10
        'Handle event from Hue button resource controller.'
        LOGGER.debug('Received button event: %s', hue_resource)
        if hue_resource.button is None:
            return
        hue_device = btn_controller.get_device(hue_resource.id)
        device = dev_reg.async_get_device(identifiers={(DOMAIN, hue_device.id)})
        data = {CONF_ID: slugify(f'{hue_device.metadata.name} Button'), CONF_DEVICE_ID: device.id, CONF_UNIQUE_ID: hue_resource.id, CONF_TYPE: hue_resource.button.last_event.value, CONF_SUBTYPE: hue_resource.metadata.control_id}
        hass.bus.async_fire(ATTR_HUE_EVENT, data)
    conf_entry.async_on_unload(btn_controller.subscribe(handle_button_event, event_filter=EventType.RESOURCE_UPDATED))

    @callback
    def handle_rotary_event(evt_type: EventType, hue_resource: RelativeRotary) -> None:
        if False:
            print('Hello World!')
        'Handle event from Hue relative_rotary resource controller.'
        LOGGER.debug('Received relative_rotary event: %s', hue_resource)
        hue_device = btn_controller.get_device(hue_resource.id)
        device = dev_reg.async_get_device(identifiers={(DOMAIN, hue_device.id)})
        data = {CONF_DEVICE_ID: device.id, CONF_UNIQUE_ID: hue_resource.id, CONF_TYPE: hue_resource.relative_rotary.last_event.action.value, CONF_SUBTYPE: hue_resource.relative_rotary.last_event.rotation.direction.value, CONF_DURATION: hue_resource.relative_rotary.last_event.rotation.duration, CONF_STEPS: hue_resource.relative_rotary.last_event.rotation.steps}
        hass.bus.async_fire(ATTR_HUE_EVENT, data)
    conf_entry.async_on_unload(rotary_controller.subscribe(handle_rotary_event, event_filter=EventType.RESOURCE_UPDATED))