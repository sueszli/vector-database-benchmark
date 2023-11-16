""" TabLayout

A ``StackLayout`` subclass that uses tabs to let the user select a child widget.

Example:

.. UIExample:: 100

    from flexx import app, ui

    class Example(ui.Widget):
        def init(self):
            with ui.TabLayout() as self.t:
                self.a = ui.Widget(title='red', style='background:#a00;')
                self.b = ui.Widget(title='green', style='background:#0a0;')
                self.c = ui.Widget(title='blue', style='background:#00a;')

Also see examples: :ref:`demo.py`.

"""
from pscript import window
from ... import event
from ._stack import StackLayout

class TabLayout(StackLayout):
    """ A StackLayout which provides a tabbar for selecting the current widget.
    The title of each child widget is used for the tab label.
    
    The ``node`` of this widget is a
    `<div> <https://developer.mozilla.org/docs/Web/HTML/Element/div>`_.
    The visible child widget fills the entire area of this element,
    except for a small area at the top where the tab-bar is shown.
    """
    CSS = '\n\n    .flx-TabLayout > .flx-Widget {\n        top: 30px;\n        margin: 0;\n        height: calc(100% - 30px);\n        border: 1px solid #ddd;\n    }\n\n    .flx-TabLayout > .flx-tabbar {\n        box-sizing: border-box;\n        position: absolute;\n        left: 0;\n        right: 0;\n        top: 0;\n        height: 30px;\n        overflow: hidden;\n    }\n\n    .flx-tabbar > .flx-tab-item {\n        display: inline-block;\n        height: 22px;  /* 100% - 8px: 3 margin + 2 borders + 2 padding -1 overlap */\n        margin-top: 3px;\n        padding: 3px 6px 1px 6px;\n\n        overflow: hidden;\n        min-width: 10px;\n\n        -webkit-user-select: none;\n        -moz-user-select: none;\n        -ms-user-select: none;\n        user-select: none;\n\n        background: #ececec;\n        border: 1px solid #bbb;\n        border-radius: 3px 3px 0px 0px;\n        margin-left: -1px;\n        transition: background 0.3s;\n    }\n    .flx-tabbar > .flx-tab-item:first-of-type {\n        margin-left: 0;\n    }\n\n    .flx-tabbar > .flx-tab-item.flx-current {\n        background: #eaecff;\n        border-top: 3px solid #7bf;\n        margin-top: 0;\n    }\n\n    .flx-tabbar > .flx-tab-item:hover {\n        background: #eaecff;\n    }\n    '

    def _create_dom(self):
        if False:
            print('Hello World!')
        outernode = window.document.createElement('div')
        self._tabbar = window.document.createElement('div')
        self._tabbar.classList.add('flx-tabbar')
        self._addEventListener(self._tabbar, 'mousedown', self._tabbar_click)
        outernode.appendChild(self._tabbar)
        return outernode

    def _render_dom(self):
        if False:
            while True:
                i = 10
        nodes = [child.outernode for child in self.children]
        nodes.append(self._tabbar)
        return nodes

    @event.reaction
    def __update_tabs(self):
        if False:
            return 10
        children = self.children
        current = self.current
        while len(self._tabbar.children) < len(children):
            node = window.document.createElement('p')
            node.classList.add('flx-tab-item')
            node.index = len(self._tabbar.children)
            self._tabbar.appendChild(node)
        while len(self._tabbar.children) > len(children):
            c = self._tabbar.children[len(self._tabbar.children) - 1]
            self._tabbar.removeChild(c)
        for i in range(len(children)):
            widget = children[i]
            node = self._tabbar.children[i]
            node.textContent = widget.title
            if widget is current:
                node.classList.add('flx-current')
            else:
                node.classList.remove('flx-current')
        self.__checks_sizes()

    @event.reaction('size')
    def __checks_sizes(self, *events):
        if False:
            return 10
        nodes = self._tabbar.children
        width = (self.size[0] - 10) / len(nodes) - 2 - 12
        for i in range(len(nodes)):
            nodes[i].style.width = width + 'px'

    @event.emitter
    def user_current(self, current):
        if False:
            return 10
        ' Event emitted when the user selects a tab. Can be used to distinguish\n        user-invoked from programatically-invoked tab changes.\n        Has ``old_value`` and ``new_value`` attributes.\n        '
        if isinstance(current, (float, int)):
            current = self.children[int(current)]
        d = {'old_value': self.current, 'new_value': current}
        self.set_current(current)
        return d

    def _tabbar_click(self, e):
        if False:
            return 10
        index = e.target.index
        if index >= 0:
            self.user_current(index)