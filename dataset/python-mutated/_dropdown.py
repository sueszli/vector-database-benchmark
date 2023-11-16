""" Dropdown widgets

.. UIExample:: 120

    from flexx import app, event, ui

    class Example(ui.Widget):

        def init(self):
            self.combo = ui.ComboBox(editable=True,
                                     options=('foo', 'bar', 'spaaaaaaaaam', 'eggs'))
            self.label = ui.Label()

        @event.reaction
        def update_label(self):
            text = 'Combobox text: ' + self.combo.text
            if self.combo.selected_index is not None:
                text += ' (index %i)' % self.combo.selected_index
            self.label.set_text(text)

Also see examples: :ref:`control_with_keys.py`.

"""
from pscript import window
from ... import event, app
from .._widget import Widget, create_element

class BaseDropdown(Widget):
    """ Base class for drop-down-like widgets.
    """
    DEFAULT_MIN_SIZE = (50, 28)
    CSS = "\n\n        .flx-BaseDropdown {\n            display: inline-block;\n            overflow: visible;\n            margin: 2px;\n            border-radius: 3px;\n            padding: 2px;\n            border: 1px solid #aaa;\n            max-height: 28px; /* overridden by maxsize */\n            white-space: nowrap; /* keep label and but on-line */\n            background: #e8e8e8\n        }\n\n        .flx-BaseDropdown:focus {\n            outline: none;\n            box-shadow: 0px 0px 3px 1px rgba(0, 100, 200, 0.7);\n        }\n\n        .flx-BaseDropdown > .flx-dd-edit {\n            display: none;\n            max-width: 2em;  /* reset silly lineedit sizing */\n            min-width: calc(100% - 1.5em - 2px);\n            min-height: 1em;\n            margin: 0;\n            padding: 0;\n            border: none;\n        }\n\n        .flx-BaseDropdown > .flx-dd-label {\n            display: inline-block;\n            min-width: calc(100% - 1.5em - 2px);\n            min-height: 1em;\n            user-select: none;\n            -moz-user-select: none;\n            -webkit-user-select: none;\n            -ms-user-select: none;\n        }\n\n        .flx-BaseDropdown.editable-true {\n            background: #fff;\n        }\n        .flx-BaseDropdown.editable-true > .flx-dd-label {\n            display: none;\n        }\n        .flx-BaseDropdown.editable-true > .flx-dd-edit {\n            display: inline-block;\n        }\n\n        .flx-BaseDropdown > .flx-dd-button {\n            display: inline-block;\n            position: static;\n            min-width: 1.5em;\n            max-width: 1.5em;\n            text-align: center;\n            margin: 0;\n        }\n        .flx-BaseDropdown > .flx-dd-button:hover {\n            background: rgba(128, 128, 128, 0.1);\n        }\n        .flx-BaseDropdown > .flx-dd-button::after {\n            content: '\\25BE';  /* 2228 1F847 1F83F */\n        }\n\n        .flx-BaseDropdown .flx-dd-space {\n            display: inline-block;\n            min-width: 1em;\n        }\n\n        .flx-BaseDropdown > .flx-dd-strud {\n            /* The strud allows to give the box a natural minimum size,\n               but it should not affect the height. */\n            visibility: hidden;\n            overflow: hidden;\n            max-height: 0;\n        }\n    "

    def init(self):
        if False:
            while True:
                i = 10
        if self.tabindex == -2:
            self.set_tabindex(-1)

    @event.action
    def expand(self):
        if False:
            for i in range(10):
                print('nop')
        ' Expand the dropdown and give it focus, so that it can be used\n        with the up/down keys.\n        '
        self._expand()
        self.node.focus()

    def _create_dom(self):
        if False:
            return 10
        return window.document.createElement('span')

    def _render_dom(self):
        if False:
            i = 10
            return i + 15
        f2 = lambda e: self._submit_text() if e.which == 13 else None
        return [create_element('span', {'className': 'flx-dd-label', 'onclick': self._but_click}, self.text + '\xa0'), create_element('input', {'className': 'flx-dd-edit', 'onkeypress': f2, 'onblur': self._submit_text, 'value': self.text}), create_element('span'), create_element('span', {'className': 'flx-dd-button', 'onclick': self._but_click}), create_element('div', {'className': 'flx-dd-strud'}, '\xa0')]

    def _but_click(self):
        if False:
            i = 10
            return i + 15
        if self.node.classList.contains('expanded'):
            self._collapse()
        else:
            self._expand()

    def _submit_text(self):
        if False:
            return 10
        edit_node = self.outernode.childNodes[1]
        self.set_text(edit_node.value)

    def _expand(self):
        if False:
            i = 10
            return i + 15
        self.node.classList.add('expanded')
        rect = self.node.getBoundingClientRect()
        self._rect_to_check = rect
        window.setTimeout(self._check_expanded_pos, 100)
        self._addEventListener(window.document, 'mousedown', self._collapse_maybe, 1)
        return rect

    def _collapse_maybe(self, e):
        if False:
            return 10
        t = e.target
        while t is not window.document.body:
            if t is self.outernode:
                return
            t = t.parentElement
        window.document.removeEventListener('mousedown', self._collapse_maybe, 1)
        self._collapse()

    def _collapse(self):
        if False:
            while True:
                i = 10
        self.node.classList.remove('expanded')

    def _check_expanded_pos(self):
        if False:
            return 10
        if self.node.classList.contains('expanded'):
            rect = self.node.getBoundingClientRect()
            if not (rect.top == self._rect_to_check.top and rect.left == self._rect_to_check.left):
                self._collapse()
            else:
                window.setTimeout(self._check_expanded_pos, 100)

class ComboBox(BaseDropdown):
    """
    The Combobox is a combination of a button and a popup list, optionally
    with an editable text. It can be used to select among a set of
    options in a more compact manner than a TreeWidget would.
    Optionally, the text of the combobox can be edited.

    It is generally good practive to react to ``user_selected`` to detect user
    interaction, and react to ``text``, ``selected_key`` or ``selected_index``
    to keep track of all kinds of (incl. programatic) interaction .

    When the combobox is expanded, the arrow keys can be used to select
    an item, and it can be made current by pressing Enter or spacebar.
    Escape can be used to collapse the combobox.

    The ``node`` of this widget is a
    `<span> <https://developer.mozilla.org/docs/Web/HTML/Element/span>`_
    with some child elements and quite a bit of CSS for rendering.
    """
    CSS = '\n\n        .flx-ComboBox {\n        }\n\n        .flx-ComboBox > ul  {\n            list-style-type: none;\n            box-sizing: border-box;\n            border: 1px solid #333;\n            border-radius: 3px;\n            margin: 0;\n            padding: 2px;\n            position: fixed;  /* because all our widgets are overflow:hidden */\n            background: white;\n            z-index: 9999;\n            display: none;\n        }\n        .flx-ComboBox.expanded > ul {\n            display: block;\n            max-height: 220px;\n            overflow-y: auto;\n        }\n\n        .flx-ComboBox.expanded > ul > li:hover {\n            background: rgba(0, 128, 255, 0.2);\n        }\n        .flx-ComboBox.expanded > ul > li.highlighted-true {\n            box-shadow: inset 0 0 3px 1px rgba(0, 0, 255, 0.4);\n        }\n    '
    text = event.StringProp('', settable=True, doc='\n        The text displayed on the widget. This property is set\n        when an item is selected from the dropdown menu. When editable,\n        the ``text`` is also set when the text is edited by the user.\n        This property is settable programatically regardless of the\n        value of ``editable``.\n        ')
    selected_index = event.IntProp(-1, settable=True, doc='\n        The currently selected item index. Can be -1 if no item has\n        been selected or when the text was changed manually (if editable).\n        Can also be programatically set.\n        ')
    selected_key = event.StringProp('', settable=True, doc="\n        The currently selected item key. Can be '' if no item has\n        been selected or when the text was changed manually (if editable).\n        Can also be programatically set.\n        ")
    placeholder_text = event.StringProp('', settable=True, doc='\n        The placeholder text to display in editable mode.\n        ')
    editable = event.BoolProp(False, settable=True, doc="\n        Whether the combobox's text is editable.\n        ")
    options = event.TupleProp((), settable=True, doc='\n        A list of tuples (key, text) representing the options. Both\n        keys and texts are converted to strings if they are not already.\n        For items that are given as a string, the key and text are the same.\n        If a dict is given, it is transformed to key-text pairs.\n        ')
    _highlighted = app.LocalProperty(-1, settable=True, doc='\n        The index of the currently highlighted item.\n        ')

    @event.action
    def set_options(self, options):
        if False:
            print('Hello World!')
        if isinstance(options, dict):
            keys = options.keys()
            keys = sorted(keys)
            options = [(k, options[k]) for k in keys]
        options2 = []
        for opt in options:
            if isinstance(opt, (tuple, list)):
                opt = (str(opt[0]), str(opt[1]))
            else:
                opt = (str(opt), str(opt))
            options2.append(opt)
        self._mutate_options(tuple(options2))
        keys = [key_text[0] for key_text in self.options]
        if self.selected_key and self.selected_key in keys:
            key = self.selected_key
            self.set_selected_key('')
            self.set_selected_key(key)
        elif 0 <= self.selected_index < len(self.options):
            index = self.selected_index
            self.set_selected_index(-1)
            self.set_selected_index(index)
        elif self.selected_key:
            self.set_selected_key('')
        else:
            pass

    def _deselect(self):
        if False:
            print('Hello World!')
        self._mutate('selected_index', -1)
        self._mutate('selected_key', '')
        if not self.editable:
            self.set_text('')

    @event.action
    def update_selected_index(self, text):
        if False:
            while True:
                i = 10
        for (index, option) in enumerate(self.options):
            if option[1] == text:
                self._mutate('selected_index', index)
                self._mutate('selected_key', option[0])
                return
        self._deselect()

    @event.action
    def set_selected_index(self, index):
        if False:
            print('Hello World!')
        if index == self.selected_index:
            return
        elif 0 <= index < len(self.options):
            (key, text) = self.options[index]
            self._mutate('selected_index', index)
            self._mutate('selected_key', key)
            self.set_text(text)
        else:
            self._deselect()

    @event.action
    def set_selected_key(self, key):
        if False:
            print('Hello World!')
        if key == self.selected_key:
            return
        elif key:
            for (index, option) in enumerate(self.options):
                if option[0] == key:
                    self._mutate('selected_index', index)
                    self._mutate('selected_key', key)
                    self.set_text(option[1])
                    return
        self._deselect()

    @event.emitter
    def user_selected(self, index):
        if False:
            return 10
        ' Event emitted when the user selects an item using the mouse or\n        keyboard. The event has attributes ``index``, ``key`` and ``text``.\n        '
        options = self.options
        if index >= 0 and index < len(options):
            (key, text) = options[index]
            self.set_selected_index(index)
            self.set_selected_key(key)
            self.set_text(text)
            return dict(index=index, key=key, text=text)

    def _create_dom(self):
        if False:
            return 10
        node = super()._create_dom()
        node.onkeydown = self._key_down
        return node

    def _render_dom(self):
        if False:
            return 10
        options = self.options
        option_nodes = []
        strud = []
        for i in range(len(options)):
            (key, text) = options[i]
            clsname = 'highlighted-true' if self._highlighted == i else ''
            li = create_element('li', dict(index=i, className=clsname), text if len(text.strip()) else '\xa0')
            strud += [text + '\xa0', create_element('span', {'class': 'flx-dd-space'}), create_element('br')]
            option_nodes.append(li)
        nodes = super()._render_dom()
        nodes[1].props.placeholder = self.placeholder_text
        nodes[-1].children = strud
        nodes.append(create_element('ul', dict(onmousedown=self._ul_click), option_nodes))
        return nodes

    @event.reaction
    def __track_editable(self):
        if False:
            i = 10
            return i + 15
        if self.editable:
            self.node.classList.remove('editable-false')
            self.node.classList.add('editable-true')
        else:
            self.node.classList.add('editable-false')
            self.node.classList.remove('editable-true')

    def _ul_click(self, e):
        if False:
            return 10
        if hasattr(e.target, 'index'):
            self._select_from_ul(e.target.index)

    def _select_from_ul(self, index):
        if False:
            for i in range(10):
                print('nop')
        self.user_selected(index)
        self._collapse()

    def _key_down(self, e):
        if False:
            return 10
        key = e.key
        if not key and e.code:
            key = e.code
        if not self.node.classList.contains('expanded'):
            if key in ['ArrowUp', 'ArrowDown']:
                e.stopPropagation()
                self.expand()
            return
        if key not in ['Escape', 'ArrowUp', 'ArrowDown', ' ', 'Enter']:
            return
        e.preventDefault()
        e.stopPropagation()
        if key == 'Escape':
            self._set_highlighted(-1)
            self._collapse()
        elif key == 'ArrowUp' or key == 'ArrowDown':
            if key == 'ArrowDown':
                hl = self._highlighted + 1
            else:
                hl = self._highlighted - 1
            self._set_highlighted(min(max(hl, 0), len(self.options) - 1))
        elif key == 'Enter' or key == ' ':
            if self._highlighted >= 0 and self._highlighted < len(self.options):
                self._select_from_ul(self._highlighted)

    def _expand(self):
        if False:
            for i in range(10):
                print('nop')
        rect = super()._expand()
        ul = self.outernode.children[len(self.outernode.children) - 1]
        ul.style.left = rect.left + 'px'
        ul.style.width = rect.width + 'px'
        ul.style.top = rect.bottom - 1 + 'px'
        space_below = window.innerHeight - rect.bottom
        if space_below < ul.clientHeight:
            space_above = rect.top
            if space_above > space_below:
                ul.style.top = rect.top - 1 - ul.clientHeight + 'px'

    def _submit_text(self):
        if False:
            return 10
        super()._submit_text()
        self.update_selected_index(self.text)

class DropdownContainer(BaseDropdown):
    """
    A dropdown widget that shows its children when expanded. This can be
    used to e.g. make a collapsable tree widget. Some styling may be required
    for the child widget to be sized appropriately.

    *Note: This widget is currently broken, because pointer events do not work in the
    contained widget (at least on Firefox).*
    """
    CSS = '\n        .flx-DropdownContainer {\n            min-width: 50px;\n        }\n        .flx-DropdownContainer > .flx-Widget {\n            position: fixed;\n            min-height: 100px;\n            max-height: 300px;\n            width: 200px;\n            background: white;\n            z-index: 10001;\n            display: none;\n        }\n        .flx-DropdownContainer.expanded > .flx-Widget {\n            display: initial;\n        }\n    '
    text = event.StringProp('', settable=True, doc='\n        The text displayed on the dropdown widget.\n        ')

    def _render_dom(self):
        if False:
            for i in range(10):
                print('nop')
        nodes = super()._render_dom()
        for widget in self.children:
            nodes.append(widget.outernode)
        return nodes

    def _expand(self):
        if False:
            i = 10
            return i + 15
        rect = super()._expand()
        node = self.children[0].outernode
        node.style.left = rect.left + 'px'
        node.style.top = rect.bottom - 1 + 'px'