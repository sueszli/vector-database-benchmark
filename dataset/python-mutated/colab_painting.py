"""
A web app that allows multiple people to colaborate in painting. People
connecting later will not see the "paint" that was added earlier. Each
person is assigned a random color which affects how that person can
best contribute to the painting.

This app might be running at the demo server: http://flexx1.zoof.io
"""
import random
from flexx import flx
COLORS = ('#eee', '#999', '#555', '#111', '#f00', '#0f0', '#00f', '#ff0', '#f0f', '#0ff', '#a44', '#4a4', '#44a', '#aa4', '#afa', '#4aa')

class Relay(flx.Component):
    """ Global object to relay paint events to all participants.
    """

    @flx.emitter
    def add_paint_for_all(self, pos, color):
        if False:
            print('Hello World!')
        return dict(pos=pos, color=color)
relay = Relay()

class ColabPainting(flx.PyWidget):
    """ The Python side of the app. There is one instance per connection.
    """
    color = flx.ColorProp(settable=True, doc='Paint color')
    status = flx.StringProp('', settable=True, doc='Status text')

    def init(self):
        if False:
            i = 10
            return i + 15
        self.set_color(random.choice(COLORS))
        self.widget = ColabPaintingView(self)
        self._update_participants()

    @flx.action
    def add_paint(self, pos):
        if False:
            return 10
        ' Add paint at the specified position.\n        '
        relay.add_paint_for_all(pos, self.color.hex)

    @relay.reaction('add_paint_for_all')
    def _any_user_adds_paint(self, *events):
        if False:
            print('Hello World!')
        ' Receive global paint event from the relay, invoke action on view.\n        '
        for ev in events:
            self.widget.add_paint_to_canvas(ev.pos, ev.color)

    @flx.manager.reaction('connections_changed')
    def _update_participants(self, *events):
        if False:
            while True:
                i = 10
        if self.session.status:
            sessions = flx.manager.get_connections(self.session.app_name)
            n = len(sessions)
            del sessions
            self.set_status('%i persons are painting' % n)

class ColabPaintingView(flx.Widget):
    """ The part of the app that runs in the browser.
    """
    CSS = '\n    .flx-ColabPaintingView { background: #ddd; }\n    .flx-ColabPaintingView .flx-CanvasWidget {\n        background: #fff;\n        border: 10px solid #000;\n    }\n    '

    def init(self, model):
        if False:
            while True:
                i = 10
        super().init()
        self.model = model
        with flx.VBox():
            flx.Label(flex=0, text=lambda : model.status)
            flx.Widget(flex=1)
            with flx.HBox(flex=2):
                flx.Widget(flex=1)
                self.canvas = flx.CanvasWidget(flex=0, minsize=400, maxsize=400)
                flx.Widget(flex=1)
            flx.Widget(flex=1)
        self._ctx = self.canvas.node.getContext('2d')

    @flx.reaction
    def __update_color(self):
        if False:
            i = 10
            return i + 15
        self.canvas.apply_style('border: 10px solid ' + self.model.color.hex)

    @flx.reaction('canvas.pointer_down')
    def __on_click(self, *events):
        if False:
            while True:
                i = 10
        for ev in events:
            self.model.add_paint(ev.pos)

    @flx.action
    def add_paint_to_canvas(self, pos, color):
        if False:
            return 10
        ' Actually draw a dot on the canvas.\n        '
        self._ctx.globalAlpha = 0.8
        self._ctx.beginPath()
        self._ctx.fillStyle = color
        self._ctx.arc(pos[0], pos[1], 5, 0, 6.2831)
        self._ctx.fill()
if __name__ == '__main__':
    a = flx.App(ColabPainting)
    a.serve()
    flx.start()