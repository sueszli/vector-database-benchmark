from flet_core.event import Event

class ControlEvent(Event):

    def __init__(self, target: str, name: str, data: str, control, page):
        if False:
            i = 10
            return i + 15
        Event.__init__(self, target=target, name=name, data=data)
        self.control = control
        self.page = page