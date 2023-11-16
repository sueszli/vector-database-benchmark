"""Manager to set up IO with Crownstone devices for a config entry."""
from __future__ import annotations
import logging
from typing import Any
from crownstone_cloud import CrownstoneCloud
from crownstone_cloud.exceptions import CrownstoneAuthenticationError, CrownstoneUnknownError
from crownstone_sse import CrownstoneSSEAsync
from crownstone_uart import CrownstoneUart, UartEventBus
from crownstone_uart.Exceptions import UartException
from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_EMAIL, CONF_PASSWORD, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.dispatcher import async_dispatcher_send
from .const import CONF_USB_PATH, CONF_USB_SPHERE, DOMAIN, PLATFORMS, PROJECT_NAME, SSE_LISTENERS, UART_LISTENERS
from .helpers import get_port
from .listeners import setup_sse_listeners, setup_uart_listeners
_LOGGER = logging.getLogger(__name__)

class CrownstoneEntryManager:
    """Manage a Crownstone config entry."""
    uart: CrownstoneUart | None = None
    cloud: CrownstoneCloud
    sse: CrownstoneSSEAsync

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize the hub.'
        self.hass = hass
        self.config_entry = config_entry
        self.listeners: dict[str, Any] = {}
        self.usb_sphere_id: str | None = None

    async def async_setup(self) -> bool:
        """Set up a Crownstone config entry.

        Returns True if the setup was successful.
        """
        email = self.config_entry.data[CONF_EMAIL]
        password = self.config_entry.data[CONF_PASSWORD]
        self.cloud = CrownstoneCloud(email=email, password=password, clientsession=aiohttp_client.async_get_clientsession(self.hass))
        try:
            await self.cloud.async_initialize()
        except CrownstoneAuthenticationError as auth_err:
            _LOGGER.error('Auth error during login with type: %s and message: %s', auth_err.type, auth_err.message)
            return False
        except CrownstoneUnknownError as unknown_err:
            _LOGGER.error('Unknown error during login')
            raise ConfigEntryNotReady from unknown_err
        self.sse = CrownstoneSSEAsync(email=email, password=password, access_token=self.cloud.access_token, websession=aiohttp_client.async_create_clientsession(self.hass), project_name=PROJECT_NAME)
        self.config_entry.async_create_background_task(self.hass, self.async_process_events(self.sse), 'crownstone-sse')
        setup_sse_listeners(self)
        if self.config_entry.options[CONF_USB_PATH] is not None:
            await self.async_setup_usb()
        self.usb_sphere_id = self.config_entry.options[CONF_USB_SPHERE]
        await self.hass.config_entries.async_forward_entry_setups(self.config_entry, PLATFORMS)
        self.config_entry.async_on_unload(self.config_entry.add_update_listener(_async_update_listener))
        self.config_entry.async_on_unload(self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, self.on_shutdown))
        return True

    async def async_process_events(self, sse_client: CrownstoneSSEAsync) -> None:
        """Asynchronous iteration of Crownstone SSE events."""
        async with sse_client as client:
            async for event in client:
                if event is not None:
                    async_dispatcher_send(self.hass, f'{DOMAIN}_{event.type}', event)

    async def async_setup_usb(self) -> None:
        """Attempt setup of a Crownstone usb dongle."""
        serial_port = await self.hass.async_add_executor_job(get_port, self.config_entry.options[CONF_USB_PATH])
        if serial_port is None:
            return
        self.uart = CrownstoneUart()
        try:
            await self.uart.initialize_usb(serial_port)
        except UartException:
            self.uart = None
            updated_options = self.config_entry.options.copy()
            updated_options[CONF_USB_PATH] = None
            updated_options[CONF_USB_SPHERE] = None
            self.hass.config_entries.async_update_entry(self.config_entry, options=updated_options)
            persistent_notification.async_create(self.hass, f'Setup of Crownstone USB dongle was unsuccessful on port {serial_port}.\n Crownstone Cloud will be used to switch Crownstones.\n Please check if your port is correct and set up the USB again from integration options.', 'Crownstone', 'crownstone_usb_dongle_setup')
            return
        setup_uart_listeners(self)

    async def async_unload(self) -> bool:
        """Unload the current config entry."""
        if self.cloud.cloud_data is None:
            return True
        self.sse.close_client()
        for sse_unsub in self.listeners[SSE_LISTENERS]:
            sse_unsub()
        if self.uart:
            self.uart.stop()
            for subscription_id in self.listeners[UART_LISTENERS]:
                UartEventBus.unsubscribe(subscription_id)
        unload_ok = await self.hass.config_entries.async_unload_platforms(self.config_entry, PLATFORMS)
        if unload_ok:
            self.hass.data[DOMAIN].pop(self.config_entry.entry_id)
        return unload_ok

    @callback
    def on_shutdown(self, _: Event) -> None:
        if False:
            print('Hello World!')
        'Close all IO connections.'
        self.sse.close_client()
        if self.uart:
            self.uart.stop()

async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)