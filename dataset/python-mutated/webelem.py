"""Generic web element related code."""
from typing import Iterator, Optional, Set, TYPE_CHECKING, Union, Dict
import collections.abc
from qutebrowser.qt import machinery
from qutebrowser.qt.core import QUrl, Qt, QEvent, QTimer, QRect, QPointF
from qutebrowser.qt.gui import QMouseEvent
from qutebrowser.config import config
from qutebrowser.keyinput import modeman
from qutebrowser.utils import log, usertypes, utils, qtutils, objreg
if TYPE_CHECKING:
    from qutebrowser.browser import browsertab
JsValueType = Union[int, float, str, None]
if machinery.IS_QT6:
    KeybordModifierType = Qt.KeyboardModifier
else:
    KeybordModifierType = Union[Qt.KeyboardModifiers, Qt.KeyboardModifier]

class Error(Exception):
    """Base class for WebElement errors."""

class OrphanedError(Error):
    """Raised when a webelement's parent has vanished."""

def css_selector(group: str, url: QUrl) -> str:
    if False:
        print('Hello World!')
    'Get a CSS selector for the given group/URL.'
    selectors = config.instance.get('hints.selectors', url)
    if group not in selectors:
        selectors = config.val.hints.selectors
        if group not in selectors:
            raise Error('Undefined hinting group {!r}'.format(group))
    return ','.join(selectors[group])

class AbstractWebElement(collections.abc.MutableMapping):
    """A wrapper around QtWebKit/QtWebEngine web element."""

    def __init__(self, tab: 'browsertab.AbstractTab') -> None:
        if False:
            print('Hello World!')
        self._tab = tab

    def __eq__(self, other: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def __getitem__(self, key: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def __setitem__(self, key: str, val: str) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def __delitem__(self, key: str) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        if False:
            return 10
        raise NotImplementedError

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        try:
            html: Optional[str] = utils.compact_text(self.outer_xml(), 500)
        except Error:
            html = None
        return utils.get_repr(self, html=html)

    def has_frame(self) -> bool:
        if False:
            print('Hello World!')
        'Check if this element has a valid frame attached.'
        raise NotImplementedError

    def geometry(self) -> QRect:
        if False:
            for i in range(10):
                print('nop')
        'Get the geometry for this element.'
        raise NotImplementedError

    def classes(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        'Get a set of classes assigned to this element.'
        raise NotImplementedError

    def tag_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get the tag name of this element.\n\n        The returned name will always be lower-case.\n        '
        raise NotImplementedError

    def outer_xml(self) -> str:
        if False:
            i = 10
            return i + 15
        'Get the full HTML representation of this element.'
        raise NotImplementedError

    def value(self) -> JsValueType:
        if False:
            for i in range(10):
                print('nop')
        'Get the value attribute for this element, or None.'
        raise NotImplementedError

    def set_value(self, value: JsValueType) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the element value.'
        raise NotImplementedError

    def dispatch_event(self, event: str, bubbles: bool=False, cancelable: bool=False, composed: bool=False) -> None:
        if False:
            return 10
        'Dispatch an event to the element.\n\n        Args:\n            event: The name of the event.\n            bubbles: Whether this event should bubble.\n            cancelable: Whether this event can be cancelled.\n            composed: Whether the event will trigger listeners outside of a\n                      shadow root.\n        '
        raise NotImplementedError

    def insert_text(self, text: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Insert the given text into the element.'
        raise NotImplementedError

    def rect_on_view(self, *, elem_geometry: QRect=None, no_js: bool=False) -> QRect:
        if False:
            while True:
                i = 10
        'Get the geometry of the element relative to the webview.\n\n        Args:\n            elem_geometry: The geometry of the element, or None.\n            no_js: Fall back to the Python implementation.\n        '
        raise NotImplementedError

    def is_writable(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check whether an element is writable.'
        return not ('disabled' in self or 'readonly' in self)

    def is_content_editable(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if an element has a contenteditable attribute.\n\n        Return:\n            True if the element has a contenteditable attribute,\n            False otherwise.\n        '
        try:
            return self['contenteditable'].lower() not in ['false', 'inherit']
        except KeyError:
            return False

    def is_content_editable_prop(self) -> bool:
        if False:
            print('Hello World!')
        'Get the value of this element\'s isContentEditable property.\n\n        The is_content_editable() method above checks for the "contenteditable"\n        HTML attribute, which does not handle inheritance. However, the actual\n        attribute value is still needed for certain cases (like strict=True).\n\n        This instead gets the isContentEditable JS property, which handles\n        inheritance.\n        '
        raise NotImplementedError

    def _is_editable_object(self) -> bool:
        if False:
            return 10
        'Check if an object-element is editable.'
        if 'type' not in self:
            log.webelem.debug('<object> without type clicked...')
            return False
        objtype = self['type'].lower()
        if objtype.startswith('application/') or 'classid' in self:
            log.webelem.debug("<object type='{}'> clicked.".format(objtype))
            return config.val.input.insert_mode.plugins
        else:
            return False

    def _is_editable_input(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if an input-element is editable.\n\n        Return:\n            True if the element is editable, False otherwise.\n        '
        try:
            objtype = self['type'].lower()
        except KeyError:
            return self.is_writable()
        else:
            if objtype in ['text', 'email', 'url', 'tel', 'number', 'password', 'search', 'date', 'time', 'datetime', 'datetime-local', 'month', 'week']:
                return self.is_writable()
            else:
                return False

    def _is_editable_classes(self) -> bool:
        if False:
            print('Hello World!')
        'Check if an element is editable based on its classes.\n\n        Return:\n            True if the element is editable, False otherwise.\n        '
        classes = {'div': ['CodeMirror', 'kix-', 'ace_'], 'pre': ['CodeMirror'], 'span': ['cm-']}
        relevant_classes = classes[self.tag_name()]
        for klass in self.classes():
            if any((klass.strip().startswith(e) for e in relevant_classes)):
                return True
        return False

    def is_editable(self, strict: bool=False) -> bool:
        if False:
            print('Hello World!')
        'Check whether we should switch to insert mode for this element.\n\n        Args:\n            strict: Whether to do stricter checking so only fields where we can\n                    get the value match, for use with the :editor command.\n\n        Return:\n            True if we should switch to insert mode, False otherwise.\n        '
        roles = ('combobox', 'textbox')
        log.webelem.debug('Checking if element is editable: {}'.format(repr(self)))
        tag = self.tag_name()
        if self.is_content_editable() and self.is_writable():
            return True
        elif self.get('role', None) in roles and self.is_writable():
            return True
        elif tag == 'input':
            return self._is_editable_input()
        elif tag == 'textarea':
            return self.is_writable()
        elif tag in ['embed', 'applet']:
            return config.val.input.insert_mode.plugins and (not strict)
        elif not strict and self.is_content_editable_prop() and self.is_writable():
            return True
        elif tag == 'object':
            return self._is_editable_object() and (not strict)
        elif tag in ['div', 'pre', 'span']:
            return self._is_editable_classes() and (not strict)
        return False

    def is_text_input(self) -> bool:
        if False:
            return 10
        'Check if this element is some kind of text box.'
        roles = ('combobox', 'textbox')
        tag = self.tag_name()
        return self.get('role', None) in roles or tag in ['input', 'textarea']

    def remove_blank_target(self) -> None:
        if False:
            print('Hello World!')
        'Remove target from link.'
        raise NotImplementedError

    def resolve_url(self, baseurl: QUrl) -> Optional[QUrl]:
        if False:
            for i in range(10):
                print('nop')
        "Resolve the URL in the element's src/href attribute.\n\n        Args:\n            baseurl: The URL to base relative URLs on as QUrl.\n\n        Return:\n            A QUrl with the absolute URL, or None.\n        "
        if baseurl.isRelative():
            raise ValueError('Need an absolute base URL!')
        for attr in ['href', 'src']:
            if attr in self:
                text = self[attr].strip()
                break
        else:
            return None
        url = QUrl(text)
        if not url.isValid():
            return None
        if url.isRelative():
            url = baseurl.resolved(url)
        qtutils.ensure_valid(url)
        return url

    def is_link(self) -> bool:
        if False:
            return 10
        'Return True if this AbstractWebElement is a link.'
        href_tags = ['a', 'area', 'link']
        return self.tag_name() in href_tags and 'href' in self

    def _requires_user_interaction(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return True if clicking this element needs user interaction.'
        raise NotImplementedError

    def _mouse_pos(self) -> QPointF:
        if False:
            i = 10
            return i + 15
        'Get the position to click/hover.'
        rect = self.rect_on_view()
        if rect.width() > rect.height():
            rect.setWidth(rect.height())
        else:
            rect.setHeight(rect.width())
        pos = rect.center()
        if pos.x() < 0 or pos.y() < 0:
            raise Error('Element position is out of view!')
        return QPointF(pos)

    def _move_text_cursor(self) -> None:
        if False:
            i = 10
            return i + 15
        'Move cursor to end after clicking.'
        raise NotImplementedError

    def _click_fake_event(self, click_target: usertypes.ClickTarget, button: Qt.MouseButton=Qt.MouseButton.LeftButton) -> None:
        if False:
            print('Hello World!')
        'Send a fake click event to the element.'
        pos = self._mouse_pos()
        log.webelem.debug('Sending fake click to {!r} at position {} with target {}'.format(self, pos, click_target))
        target_modifiers: Dict[usertypes.ClickTarget, KeybordModifierType] = {usertypes.ClickTarget.normal: Qt.KeyboardModifier.NoModifier, usertypes.ClickTarget.window: Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ShiftModifier, usertypes.ClickTarget.tab: Qt.KeyboardModifier.ControlModifier, usertypes.ClickTarget.tab_bg: Qt.KeyboardModifier.ControlModifier}
        if config.val.tabs.background:
            target_modifiers[usertypes.ClickTarget.tab] |= Qt.KeyboardModifier.ShiftModifier
        else:
            target_modifiers[usertypes.ClickTarget.tab_bg] |= Qt.KeyboardModifier.ShiftModifier
        modifiers = target_modifiers[click_target]
        events = [QMouseEvent(QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier), QMouseEvent(QEvent.Type.MouseButtonPress, pos, button, button, modifiers), QMouseEvent(QEvent.Type.MouseButtonRelease, pos, button, Qt.MouseButton.NoButton, modifiers)]
        for evt in events:
            self._tab.send_event(evt)
        QTimer.singleShot(0, self._move_text_cursor)

    def _click_editable(self, click_target: usertypes.ClickTarget) -> None:
        if False:
            while True:
                i = 10
        'Fake a click on an editable input field.'
        raise NotImplementedError

    def _click_js(self, click_target: usertypes.ClickTarget) -> None:
        if False:
            i = 10
            return i + 15
        'Fake a click by using the JS .click() method.'
        raise NotImplementedError

    def delete(self) -> None:
        if False:
            return 10
        'Delete this element from the DOM.'
        raise NotImplementedError

    def _click_href(self, click_target: usertypes.ClickTarget) -> None:
        if False:
            return 10
        'Fake a click on an element with a href by opening the link.'
        baseurl = self._tab.url()
        url = self.resolve_url(baseurl)
        if url is None:
            self._click_fake_event(click_target)
            return
        tabbed_browser = objreg.get('tabbed-browser', scope='window', window=self._tab.win_id)
        if click_target in [usertypes.ClickTarget.tab, usertypes.ClickTarget.tab_bg]:
            background = click_target == usertypes.ClickTarget.tab_bg
            tabbed_browser.tabopen(url, background=background)
        elif click_target == usertypes.ClickTarget.window:
            from qutebrowser.mainwindow import mainwindow
            window = mainwindow.MainWindow(private=tabbed_browser.is_private)
            window.tabbed_browser.tabopen(url)
            window.show()
        else:
            raise ValueError('Unknown ClickTarget {}'.format(click_target))

    def click(self, click_target: usertypes.ClickTarget, *, force_event: bool=False) -> None:
        if False:
            print('Hello World!')
        'Simulate a click on the element.\n\n        Args:\n            click_target: A usertypes.ClickTarget member, what kind of click\n                          to simulate.\n            force_event: Force generating a fake mouse event.\n        '
        log.webelem.debug('Clicking {!r} with click_target {}, force_event {}'.format(self, click_target, force_event))
        if force_event:
            self._click_fake_event(click_target)
            return
        if click_target == usertypes.ClickTarget.normal:
            if self.is_link() and (not self._requires_user_interaction()):
                log.webelem.debug('Clicking via JS click()')
                self._click_js(click_target)
            elif self.is_editable(strict=True):
                log.webelem.debug('Clicking via JS focus()')
                self._click_editable(click_target)
                if config.val.input.insert_mode.auto_enter:
                    modeman.enter(self._tab.win_id, usertypes.KeyMode.insert, 'clicking input')
            else:
                self._click_fake_event(click_target)
        elif click_target in [usertypes.ClickTarget.tab, usertypes.ClickTarget.tab_bg, usertypes.ClickTarget.window]:
            if self.is_link():
                self._click_href(click_target)
            else:
                self._click_fake_event(click_target)
        else:
            raise ValueError('Unknown ClickTarget {}'.format(click_target))

    def hover(self) -> None:
        if False:
            while True:
                i = 10
        'Simulate a mouse hover over the element.'
        pos = self._mouse_pos()
        event = QMouseEvent(QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        self._tab.send_event(event)

    def right_click(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Simulate a right-click on the element.'
        self._click_fake_event(usertypes.ClickTarget.normal, button=Qt.MouseButton.RightButton)