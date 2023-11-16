"""Push notification handling."""
from __future__ import annotations
import asyncio
from collections.abc import Callable
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_call_later
from homeassistant.util.uuid import random_uuid_hex
PUSH_CONFIRM_TIMEOUT = 10

class PushChannel:
    """Class that represents a push channel."""

    def __init__(self, hass: HomeAssistant, webhook_id: str, support_confirm: bool, send_message: Callable[[dict], None], on_teardown: Callable[[], None]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize a local push channel.'
        self.hass = hass
        self.webhook_id = webhook_id
        self.support_confirm = support_confirm
        self._send_message = send_message
        self.on_teardown = on_teardown
        self.pending_confirms: dict[str, dict] = {}

    @callback
    def async_send_notification(self, data, fallback_send):
        if False:
            i = 10
            return i + 15
        'Send a push notification.'
        if not self.support_confirm:
            self._send_message(data)
            return
        confirm_id = random_uuid_hex()
        data['hass_confirm_id'] = confirm_id

        async def handle_push_failed(_=None):
            """Handle a failed local push notification."""
            if self.pending_confirms.pop(confirm_id, None) is None:
                return
            if self.on_teardown is not None:
                await self.async_teardown()
            await fallback_send(data)
        self.pending_confirms[confirm_id] = {'unsub_scheduled_push_failed': async_call_later(self.hass, PUSH_CONFIRM_TIMEOUT, handle_push_failed), 'handle_push_failed': handle_push_failed}
        self._send_message(data)

    @callback
    def async_confirm_notification(self, confirm_id) -> bool:
        if False:
            return 10
        'Confirm a push notification.\n\n        Returns if confirmation successful.\n        '
        if confirm_id not in self.pending_confirms:
            return False
        self.pending_confirms.pop(confirm_id)['unsub_scheduled_push_failed']()
        return True

    async def async_teardown(self):
        """Tear down this channel."""
        if self.on_teardown is None:
            return
        self.on_teardown()
        self.on_teardown = None
        cancel_pending_local_tasks = [actions['handle_push_failed']() for actions in self.pending_confirms.values()]
        if cancel_pending_local_tasks:
            await asyncio.gather(*cancel_pending_local_tasks)