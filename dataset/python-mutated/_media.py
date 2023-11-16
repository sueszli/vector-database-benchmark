""" Media widgets

.. UIExample:: 200

    from flexx import ui

    class Example(ui.Widget):

        def init(self):
            with ui.HSplit():
                url = 'http://www.w3schools.com/tags/mov_bbb.mp4'
                ui.VideoWidget(source=url)
                ui.YoutubeWidget(source='RG1P8MQS1cU')
                with ui.VBox():
                    stretch = ui.CheckBox(text='Stretch')
                    ui.ImageWidget(flex=1, stretch=lambda:stretch.checked,
                                    source='http://github.com/fluidicon.png')

"""
from ... import event
from . import Widget

class ImageWidget(Widget):
    """ Display an image from a url.
    
    The ``node`` of this widget is an
    `<img> <https://developer.mozilla.org/docs/Web/HTML/Element/img>`_
    wrapped in a `<div> <https://developer.mozilla.org/docs/Web/HTML/Element/div>`_
    (the ``outernode``) to handle sizing.
    """
    DEFAULT_MIN_SIZE = (16, 16)
    _sequence = 0
    source = event.StringProp('', settable=True, doc='\n        The source of the image, This can be anything that an HTML\n        img element supports.\n        ')
    stretch = event.BoolProp(False, settable=True, doc='\n        Whether the image should stretch to fill all available\n        space, or maintain its aspect ratio (default).\n        ')

    def _create_dom(self):
        if False:
            print('Hello World!')
        global window
        outer = window.document.createElement('div')
        inner = window.document.createElement('img')
        outer.appendChild(inner)
        return (outer, inner)

    @event.reaction
    def __resize_image(self):
        if False:
            print('Hello World!')
        size = self.size
        if self.stretch:
            self.node.style.maxWidth = None
            self.node.style.maxHeight = None
            self.node.style.width = size[0] + 'px'
            self.node.style.height = size[1] + 'px'
        else:
            self.node.style.maxWidth = size[0] + 'px'
            self.node.style.maxHeight = size[1] + 'px'
            self.node.style.width = None
            self.node.style.height = None

    @event.reaction
    def __source_changed(self):
        if False:
            i = 10
            return i + 15
        self.node.src = self.source

class VideoWidget(Widget):
    """ Display a video from a url.
    
    The ``node`` of this widget is a
    `<video> <https://developer.mozilla.org/docs/Web/HTML/Element/video>`_.
    """
    DEFAULT_MIN_SIZE = (100, 100)
    source = event.StringProp('', settable=True, doc='\n        The source of the video. This must be a url of a resource\n        on the web.\n        ')

    def _create_dom(self):
        if False:
            print('Hello World!')
        global window
        node = window.document.createElement('video')
        node.controls = 'controls'
        node.textContent = 'Your browser does not support HTML5 video.'
        self.src_node = window.document.createElement('source')
        self.src_node.type = 'video/mp4'
        self.src_node.src = None
        node.appendChild(self.src_node)
        return node

    def _render_dom(self):
        if False:
            print('Hello World!')
        return None

    @event.reaction
    def __source_changed(self):
        if False:
            print('Hello World!')
        self.src_node.src = self.source or None
        self.node.load()

class YoutubeWidget(Widget):
    """ Display a Youtube video.
    
    The ``node`` of this widget is a
    `<div> <https://developer.mozilla.org/docs/Web/HTML/Element/div>`_
    containing an
    `<iframe> <https://developer.mozilla.org/docs/Web/HTML/Element/iframe>`_
    that loads the youtube page.
    """
    DEFAULT_MIN_SIZE = (100, 100)
    source = event.StringProp('oHg5SJYRHA0', settable=True, doc='\n        The source of the video represented as the Youtube id.\n        ')

    def _create_dom(self):
        if False:
            for i in range(10):
                print('nop')
        global window
        node = window.document.createElement('div')
        self.inode = window.document.createElement('iframe')
        node.appendChild(self.inode)
        return node

    @event.reaction
    def _update_canvas_size(self, *events):
        if False:
            for i in range(10):
                print('nop')
        size = self.size
        if size[0] or size[1]:
            self.inode.style.width = size[0] + 'px'
            self.inode.style.height = size[1] + 'px'

    @event.reaction
    def __source_changed(self, *events):
        if False:
            while True:
                i = 10
        base_url = 'http://www.youtube.com/embed/'
        self.inode.src = base_url + self.source + '?autoplay=0'