"""QtWebEngine specific part of the web element API."""
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Optional, Set, Tuple, Union
from qutebrowser.qt.core import QRect, QEventLoop
from qutebrowser.qt.widgets import QApplication
from qutebrowser.qt.webenginecore import QWebEngineSettings
from qutebrowser.utils import log, javascript, urlutils, usertypes, utils, version
from qutebrowser.browser import webelem
if TYPE_CHECKING:
    from qutebrowser.browser.webengine import webenginetab

class WebEngineElement(webelem.AbstractWebElement):
    """A web element for QtWebEngine, using JS under the hood."""
    _tab: 'webenginetab.WebEngineTab'

    def __init__(self, js_dict: Dict[str, Any], tab: 'webenginetab.WebEngineTab') -> None:
        if False:
            print('Hello World!')
        super().__init__(tab)
        js_dict_types: Dict[str, Union[type, Tuple[type, ...]]] = {'id': int, 'text': str, 'value': (str, int, float), 'tag_name': str, 'outer_xml': str, 'class_name': str, 'rects': list, 'attributes': dict, 'is_content_editable': bool, 'caret_position': (int, type(None))}
        assert set(js_dict.keys()).issubset(js_dict_types.keys())
        for (name, typ) in js_dict_types.items():
            if name in js_dict and (not isinstance(js_dict[name], typ)):
                raise TypeError('Got {} for {} from JS but expected {}: {}'.format(type(js_dict[name]), name, typ, js_dict))
        for (name, value) in js_dict['attributes'].items():
            if not isinstance(name, str):
                raise TypeError('Got {} ({}) for attribute name from JS: {}'.format(name, type(name), js_dict))
            if not isinstance(value, str):
                raise TypeError('Got {} ({}) for attribute {} from JS: {}'.format(value, type(value), name, js_dict))
        for rect in js_dict['rects']:
            assert set(rect.keys()) == {'top', 'right', 'bottom', 'left', 'height', 'width'}, rect.keys()
            for value in rect.values():
                if not isinstance(value, (int, float)):
                    raise TypeError('Got {} ({}) for rect from JS: {}'.format(value, type(value), js_dict))
        self._id = js_dict['id']
        self._js_dict = js_dict

    def __str__(self) -> str:
        if False:
            return 10
        return self._js_dict.get('text', '')

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        if not isinstance(other, WebEngineElement):
            return NotImplemented
        return self._id == other._id

    def __getitem__(self, key: str) -> str:
        if False:
            i = 10
            return i + 15
        attrs = self._js_dict['attributes']
        return attrs[key]

    def __setitem__(self, key: str, val: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._js_dict['attributes'][key] = val
        self._js_call('set_attribute', key, val)

    def __delitem__(self, key: str) -> None:
        if False:
            while True:
                i = 10
        utils.unused(key)
        log.stub()

    def __iter__(self) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        return iter(self._js_dict['attributes'])

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self._js_dict['attributes'])

    def _js_call(self, name: str, *args: webelem.JsValueType, callback: Callable[[Any], None]=None) -> None:
        if False:
            print('Hello World!')
        'Wrapper to run stuff from webelem.js.'
        if self._tab.is_deleted():
            raise webelem.OrphanedError('Tab containing element vanished')
        js_code = javascript.assemble('webelem', name, self._id, *args)
        self._tab.run_js_async(js_code, callback=callback)

    def has_frame(self) -> bool:
        if False:
            print('Hello World!')
        return True

    def geometry(self) -> QRect:
        if False:
            while True:
                i = 10
        log.stub()
        return QRect()

    def classes(self) -> Set[str]:
        if False:
            while True:
                i = 10
        'Get a list of classes assigned to this element.'
        return set(self._js_dict['class_name'].split())

    def tag_name(self) -> str:
        if False:
            i = 10
            return i + 15
        'Get the tag name of this element.\n\n        The returned name will always be lower-case.\n        '
        tag = self._js_dict['tag_name']
        assert isinstance(tag, str), tag
        return tag.lower()

    def outer_xml(self) -> str:
        if False:
            return 10
        'Get the full HTML representation of this element.'
        return self._js_dict['outer_xml']

    def is_content_editable_prop(self) -> bool:
        if False:
            while True:
                i = 10
        return self._js_dict['is_content_editable']

    def value(self) -> webelem.JsValueType:
        if False:
            i = 10
            return i + 15
        return self._js_dict.get('value', None)

    def set_value(self, value: webelem.JsValueType) -> None:
        if False:
            while True:
                i = 10
        self._js_call('set_value', value)

    def dispatch_event(self, event: str, bubbles: bool=False, cancelable: bool=False, composed: bool=False) -> None:
        if False:
            return 10
        self._js_call('dispatch_event', event, bubbles, cancelable, composed)

    def caret_position(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        'Get the text caret position for the current element.\n\n        If the element is not a text element, None is returned.\n        '
        return self._js_dict.get('caret_position', None)

    def insert_text(self, text: str) -> None:
        if False:
            while True:
                i = 10
        if not self.is_editable(strict=True):
            raise webelem.Error('Element is not editable!')
        log.webelem.debug('Inserting text into element {!r}'.format(self))
        self._js_call('insert_text', text)

    def rect_on_view(self, *, elem_geometry: QRect=None, no_js: bool=False) -> QRect:
        if False:
            return 10
        'Get the geometry of the element relative to the webview.\n\n        Skipping of small rectangles is due to <a> elements containing other\n        elements with "display:block" style, see\n        https://github.com/qutebrowser/qutebrowser/issues/1298\n\n        Args:\n            elem_geometry: The geometry of the element, or None.\n                           Ignored with QtWebEngine.\n            no_js: Fall back to the Python implementation.\n                   Ignored with QtWebEngine.\n        '
        utils.unused(elem_geometry)
        utils.unused(no_js)
        rects = self._js_dict['rects']
        for rect in rects:
            width = rect['width']
            height = rect['height']
            left = rect['left']
            top = rect['top']
            if width > 1 and height > 1:
                zoom = self._tab.zoom.factor()
                rect = QRect(int(left * zoom), int(top * zoom), int(width * zoom), int(height * zoom))
                return rect
        log.webelem.debug("Couldn't find rectangle for {!r} ({})".format(self, rects))
        return QRect()

    def remove_blank_target(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._js_dict['attributes'].get('target') == '_blank':
            self._js_dict['attributes']['target'] = '_top'
        self._js_call('remove_blank_target')

    def delete(self) -> None:
        if False:
            while True:
                i = 10
        self._js_call('delete')

    def _move_text_cursor(self) -> None:
        if False:
            return 10
        if self.is_text_input() and self.is_editable():
            self._js_call('move_cursor_to_end')

    def _requires_user_interaction(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        baseurl = self._tab.url()
        url = self.resolve_url(baseurl)
        if url is None:
            return True
        if baseurl.scheme() == url.scheme():
            return False
        versions = version.qtwebengine_versions()
        for scheme in ['qute', 'file']:
            if baseurl.scheme() == scheme and url.scheme() != scheme and (versions.webengine >= utils.VersionNumber(6, 3)):
                return True
        return url.scheme() not in urlutils.WEBENGINE_SCHEMES

    def _click_editable(self, click_target: usertypes.ClickTarget) -> None:
        if False:
            while True:
                i = 10
        self._js_call('focus')
        self._move_text_cursor()

    def _click_js(self, _click_target: usertypes.ClickTarget) -> None:
        if False:
            i = 10
            return i + 15
        view = self._tab._widget
        assert view is not None
        attribute = QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows
        could_open_windows = view.settings().testAttribute(attribute)
        view.settings().setAttribute(attribute, True)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeSocketNotifiers | QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        def reset_setting(_arg: Any) -> None:
            if False:
                i = 10
                return i + 15
            'Set the JavascriptCanOpenWindows setting to its old value.'
            assert view is not None
            try:
                view.settings().setAttribute(attribute, could_open_windows)
            except RuntimeError:
                pass
        self._js_call('click', callback=reset_setting)