""" CanvasWidget

The canvas can be used for specialized graphics of many sorts. It can
provide either a WebGL context or a 2d context as in the example below:

.. UIExample:: 100

    from flexx import app, event, ui

    class Example(ui.CanvasWidget):

        def init(self):
            super().init()
            self.ctx = self.node.getContext('2d')
            self.set_capture_mouse(1)
            self._last_pos = (0, 0)

        @event.reaction('pointer_move')
        def on_move(self, *events):
            for ev in events:
                self.ctx.beginPath()
                self.ctx.strokeStyle = '#080'
                self.ctx.lineWidth = 3
                self.ctx.lineCap = 'round'
                self.ctx.moveTo(*self._last_pos)
                self.ctx.lineTo(*ev.pos)
                self.ctx.stroke()
                self._last_pos = ev.pos

        @event.reaction('pointer_down')
        def on_down(self, *events):
            self._last_pos = events[-1].pos

Also see example: :ref:`drawing.py`, :ref:`splines.py`.

"""
from ... import event
from .. import Widget
perf_counter = None

class CanvasWidget(Widget):
    """ A widget that provides an HTML5 canvas. The canvas is scaled with
    the available space. Use ``self.node.getContext('2d')`` or
    ``self.node.getContext('webgl')`` in the ``init()`` method to get
    a contex to perform the actual drawing.
    
    The ``node`` of this widget is a
    `<canvas> <https://developer.mozilla.org/docs/Web/HTML/Element/canvas>`_
    wrapped in a `<div> <https://developer.mozilla.org/docs/Web/HTML/Element/div>`_
    (the ``outernode``) to handle sizing.
    """
    DEFAULT_MIN_SIZE = (50, 50)
    CSS = '\n    .flx-CanvasWidget {\n        -webkit-user-select: none;\n        -moz-user-select: none;\n        -ms-user-select: none;\n        user-select: none;\n    }\n    .flx-CanvasWidget > canvas {\n        /* Set position to absolute so that the canvas is not going\n         * to be forcing a size on the container div. */\n        position: absolute;\n    }\n    '
    capture_wheel = event.BoolProp(False, settable=True, doc='\n        Whether the wheel event is "captured", i.e. not propagated to result\n        into scrolling of the parent widget (or page). If True, if no scrolling\n        must have been performed outside of the widget for about half a second\n        in order for the widget to capture scroll events.\n        ')

    def _create_dom(self):
        if False:
            for i in range(10):
                print('nop')
        global window
        outernode = window.document.createElement('div')
        innernode = window.document.createElement('canvas')
        innernode.id = self.id + '-canvas'
        outernode.appendChild(innernode)
        for ev_name in ('contextmenu', 'click', 'dblclick'):
            self._addEventListener(window.document, ev_name, self._prevent_default_event, 0)

        def wheel_behavior(e):
            if False:
                for i in range(10):
                    print('nop')
            (id, t0) = window.flexx._wheel_timestamp
            t1 = perf_counter()
            if t1 - t0 < 0.5:
                window.flexx._wheel_timestamp = (id, t1)
            else:
                window.flexx._wheel_timestamp = (e.target.id, t1)
        if not window.flexx._wheel_timestamp:
            window.flexx._wheel_timestamp = (0, '')
            self._addEventListener(window.document, 'wheel', wheel_behavior, 0)
        return (outernode, innernode)

    def _prevent_default_event(self, e):
        if False:
            for i in range(10):
                print('nop')
        ' Prevent the default action of an event unless all modifier\n        keys (shift, ctrl, alt) are pressed down.\n        '
        if e.target is self.node:
            if not (e.altKey is True and e.ctrlKey is True and (e.shiftKey is True)):
                e.preventDefault()

    def _create_pointer_event(self, e):
        if False:
            i = 10
            return i + 15
        if e.type.startswith('touch'):
            e.preventDefault()
        return super()._create_pointer_event(e)

    @event.emitter
    def pointer_wheel(self, e):
        if False:
            i = 10
            return i + 15
        global window
        if self.capture_wheel <= 0:
            return super().pointer_wheel(e)
        elif window.flexx._wheel_timestamp[0] == self.node.id:
            e.preventDefault()
            return super().pointer_wheel(e)

    @event.reaction
    def _update_canvas_size(self, *events):
        if False:
            print('Hello World!')
        size = self.size
        if size[0] or size[1]:
            self.node.width = size[0]
            self.node.height = size[1]
            self.node.style.width = size[0] + 'px'
            self.node.style.height = size[1] + 'px'