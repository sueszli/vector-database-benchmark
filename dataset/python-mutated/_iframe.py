""" IFrame

.. UIExample:: 100

    with ui.HSplit():
        ui.IFrame(url='bsdf.io')
        ui.IFrame(url='http://flexx.readthedocs.io')
        # Note: the rtd page does not seem to load on Firefox 57.04

"""
from ... import event
from . import Widget

class IFrame(Widget):
    """ An iframe element, i.e. a container to show web-content.
    Note that some websites do not allow themselves to be rendered in
    a cross-source iframe.
    
    The ``node`` of this widget is a
    `<iframe> <https://developer.mozilla.org/docs/Web/HTML/Element/iframe>`_. 
    """
    DEFAULT_MIN_SIZE = (10, 10)
    CSS = '\n        .flx-IFrame {\n            border: none;\n        }\n    '
    url = event.StringProp('', settable=True, doc="\n        The url to show. 'http://' is automatically prepended if the url\n        does not have '://' in it.\n        ")

    def _create_dom(self):
        if False:
            while True:
                i = 10
        global document
        return document.createElement('iframe')

    @event.reaction('size')
    def __on_size(self, *events):
        if False:
            print('Hello World!')
        self.node.width = self.size[0]

    @event.reaction('url')
    def _update_url(self, *events):
        if False:
            print('Hello World!')
        url = self.url
        if url and '://' not in url:
            url = 'http://' + url
        self.node.src = url