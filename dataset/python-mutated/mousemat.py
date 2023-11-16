from openrazer.client.devices import RazerDevice as __RazerDevice

class RazerMousemat(__RazerDevice):

    def trigger_reactive(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Trigger a reactive flash\n        '
        return self._dbus_interfaces['device'].triggerReactive()