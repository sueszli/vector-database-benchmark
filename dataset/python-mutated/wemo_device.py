"""Home Assistant wrapper for a pyWeMo device."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, fields
from datetime import timedelta
import logging
from typing import Literal
from pywemo import Insight, LongPressMixin, WeMoDevice
from pywemo.exceptions import ActionException, PyWeMoException
from pywemo.subscribe import EVENT_TYPE_LONG_PRESS, SubscriptionRegistry
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_CONFIGURATION_URL, ATTR_IDENTIFIERS, CONF_DEVICE_ID, CONF_NAME, CONF_PARAMS, CONF_TYPE, CONF_UNIQUE_ID
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import CONNECTION_UPNP, DeviceInfo, async_get as async_get_device_registry
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from .const import DOMAIN, WEMO_SUBSCRIPTION_EVENT
from .models import async_wemo_data
_LOGGER = logging.getLogger(__name__)
ErrorStringKey = Literal['long_press_requires_subscription']
OptionsFieldKey = Literal['enable_subscription', 'enable_long_press']

class OptionsValidationError(Exception):
    """Error validating options."""

    def __init__(self, field_key: OptionsFieldKey, error_key: ErrorStringKey, message: str) -> None:
        if False:
            print('Hello World!')
        'Store field and error_key so the exception handler can used them.\n\n        The field_key and error_key strings must be the same as in strings.json.\n\n        Args:\n          field_key: Name of the options.step.init.data key that corresponds to this error.\n            field_key must also match one of the field names inside the Options class.\n          error_key: Name of the options.error key that corresponds to this error.\n          message: Message for the Exception class.\n        '
        super().__init__(message)
        self.field_key = field_key
        self.error_key = error_key

@dataclass(frozen=True)
class Options:
    """Configuration options for the DeviceCoordinator class.

    Note: The field names must match the keys (OptionsFieldKey)
    from options.step.init.data in strings.json.
    """
    enable_subscription: bool = True
    enable_long_press: bool = True

    def __post_init__(self) -> None:
        if False:
            while True:
                i = 10
        'Validate parameters.'
        if not self.enable_subscription and self.enable_long_press:
            raise OptionsValidationError('enable_subscription', 'long_press_requires_subscription', 'Local push update subscriptions must be enabled to use long-press events')

class DeviceCoordinator(DataUpdateCoordinator[None]):
    """Home Assistant wrapper for a pyWeMo device."""
    options: Options | None = None

    def __init__(self, hass: HomeAssistant, wemo: WeMoDevice, device_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize DeviceCoordinator.'
        super().__init__(hass, _LOGGER, name=wemo.name, update_interval=timedelta(seconds=30))
        self.hass = hass
        self.wemo = wemo
        self.device_id = device_id
        self.device_info = _create_device_info(wemo)
        self.supports_long_press = isinstance(wemo, LongPressMixin)
        self.update_lock = asyncio.Lock()

    def subscription_callback(self, _device: WeMoDevice, event_type: str, params: str) -> None:
        if False:
            print('Hello World!')
        'Receives push notifications from WeMo devices.'
        _LOGGER.debug('Subscription event (%s) for %s', event_type, self.wemo.name)
        if event_type == EVENT_TYPE_LONG_PRESS:
            self.hass.bus.fire(WEMO_SUBSCRIPTION_EVENT, {CONF_DEVICE_ID: self.device_id, CONF_NAME: self.wemo.name, CONF_TYPE: event_type, CONF_PARAMS: params, CONF_UNIQUE_ID: self.wemo.serial_number})
        else:
            updated = self.wemo.subscription_update(event_type, params)
            self.hass.create_task(self._async_subscription_callback(updated))

    async def async_shutdown(self) -> None:
        """Unregister push subscriptions and remove from coordinators dict."""
        await super().async_shutdown()
        del _async_coordinators(self.hass)[self.device_id]
        assert self.options
        if self.options.enable_subscription:
            await self._async_set_enable_subscription(False)
        if self.options.enable_long_press and self.last_update_success:
            await self._async_set_enable_long_press(False)

    async def _async_set_enable_subscription(self, enable_subscription: bool) -> None:
        """Turn on/off push updates from the device."""
        registry = _async_registry(self.hass)
        if enable_subscription:
            registry.on(self.wemo, None, self.subscription_callback)
            await self.hass.async_add_executor_job(registry.register, self.wemo)
        elif self.options is not None:
            await self.hass.async_add_executor_job(registry.unregister, self.wemo)

    async def _async_set_enable_long_press(self, enable_long_press: bool) -> None:
        """Turn on/off long-press events from the device."""
        if not (isinstance(self.wemo, LongPressMixin) and self.supports_long_press):
            return
        try:
            if enable_long_press:
                await self.hass.async_add_executor_job(self.wemo.ensure_long_press_virtual_device)
            elif self.options is not None:
                await self.hass.async_add_executor_job(self.wemo.remove_long_press_virtual_device)
        except PyWeMoException:
            _LOGGER.exception('Failed to enable long press support for device: %s', self.wemo.name)
            self.supports_long_press = False

    async def async_set_options(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Update the configuration options for the device."""
        options = Options(**config_entry.options)
        _LOGGER.debug('async_set_options old(%s) new(%s)', repr(self.options), repr(options))
        for field in fields(options):
            new_value = getattr(options, field.name)
            if self.options is None or getattr(self.options, field.name) != new_value:
                await getattr(self, f'_async_set_{field.name}')(new_value)
        self.options = options

    async def _async_subscription_callback(self, updated: bool) -> None:
        """Update the state by the Wemo device."""
        if self.update_lock.locked():
            return
        try:
            await self._async_locked_update(not updated)
        except UpdateFailed as err:
            self.last_exception = err
            if self.last_update_success:
                _LOGGER.exception('Subscription callback failed')
                self.last_update_success = False
        except Exception as err:
            self.last_exception = err
            self.last_update_success = False
            _LOGGER.exception('Unexpected error fetching %s data: %s', self.name, err)
        else:
            self.async_set_updated_data(None)

    @property
    def should_poll(self) -> bool:
        if False:
            print('Hello World!')
        'Return True if polling is needed to update the state for the device.\n\n        The alternative, when this returns False, is to rely on the subscription\n        "push updates" to update the device state in Home Assistant.\n        '
        if isinstance(self.wemo, Insight) and self.wemo.get_state() == 0:
            return True
        return not (_async_registry(self.hass).is_subscribed(self.wemo) and self.last_update_success)

    async def _async_update_data(self) -> None:
        """Update WeMo state."""
        if not self.should_poll:
            return
        if self.update_lock.locked():
            return
        await self._async_locked_update(True)

    async def _async_locked_update(self, force_update: bool) -> None:
        """Try updating within an async lock."""
        async with self.update_lock:
            try:
                await self.hass.async_add_executor_job(self.wemo.get_state, force_update)
            except ActionException as err:
                raise UpdateFailed('WeMo update failed') from err

def _create_device_info(wemo: WeMoDevice) -> DeviceInfo:
    if False:
        print('Hello World!')
    'Create device information. Modify if special device.'
    _dev_info = _device_info(wemo)
    if wemo.model_name == 'DLI emulated Belkin Socket':
        _dev_info[ATTR_CONFIGURATION_URL] = f'http://{wemo.host}'
        _dev_info[ATTR_IDENTIFIERS] = {(DOMAIN, wemo.serial_number[:-1])}
    return _dev_info

def _device_info(wemo: WeMoDevice) -> DeviceInfo:
    if False:
        return 10
    return DeviceInfo(connections={(CONNECTION_UPNP, wemo.udn)}, identifiers={(DOMAIN, wemo.serial_number)}, manufacturer='Belkin', model=wemo.model_name, name=wemo.name, sw_version=wemo.firmware_version)

async def async_register_device(hass: HomeAssistant, config_entry: ConfigEntry, wemo: WeMoDevice) -> DeviceCoordinator:
    """Register a device with home assistant and enable pywemo event callbacks."""
    await hass.async_add_executor_job(wemo.get_state, True)
    device_registry = async_get_device_registry(hass)
    entry = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, **_create_device_info(wemo))
    device = DeviceCoordinator(hass, wemo, entry.id)
    _async_coordinators(hass)[entry.id] = device
    config_entry.async_on_unload(config_entry.add_update_listener(device.async_set_options))
    await device.async_set_options(hass, config_entry)
    return device

@callback
def async_get_coordinator(hass: HomeAssistant, device_id: str) -> DeviceCoordinator:
    if False:
        i = 10
        return i + 15
    'Return DeviceCoordinator for device_id.'
    return _async_coordinators(hass)[device_id]

@callback
def _async_coordinators(hass: HomeAssistant) -> dict[str, DeviceCoordinator]:
    if False:
        i = 10
        return i + 15
    return async_wemo_data(hass).config_entry_data.device_coordinators

@callback
def _async_registry(hass: HomeAssistant) -> SubscriptionRegistry:
    if False:
        for i in range(10):
            print('nop')
    return async_wemo_data(hass).registry