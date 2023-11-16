from __future__ import absolute_import
import abc
import functools
import io
import json
import logging
import re
import threading
import time
from collections import defaultdict
from types import ModuleType
from typing import Callable, Optional, Union
import adbutils
from deprecated import deprecated
from logzero import logger, setup_logger
from PIL import Image
import uiautomator2
from ._proto import Direction
from .abcd import BasicUIMeta
from .exceptions import XPathElementNotFoundError
from .utils import U, inject_call, swipe_in_bounds
try:
    from lxml import etree
except ImportError:
    logger.warning('lxml was not installed, xpath will not supported')

def safe_xmlstr(s):
    if False:
        while True:
            i = 10
    return s.replace('$', '-')

def init():
    if False:
        while True:
            i = 10
    uiautomator2.plugin_register('xpath', XPath)

def string_quote(s):
    if False:
        i = 10
        return i + 15
    ' quick way to quote string '
    return '{!r}'.format(s)

def str2bytes(v) -> bytes:
    if False:
        i = 10
        return i + 15
    if isinstance(v, bytes):
        return v
    return v.encode('utf-8')

def strict_xpath(xpath: str, logger=logger) -> str:
    if False:
        while True:
            i = 10
    ' make xpath to be computer recognized xpath '
    orig_xpath = xpath
    if xpath.startswith('/'):
        pass
    elif xpath.startswith('@'):
        xpath = '//*[@resource-id={!r}]'.format(xpath[1:])
    elif xpath.startswith('^'):
        xpath = '//*[re:match(@text, {0}) or re:match(@content-desc, {0}) or re:match(@resource-id, {0})]'.format(string_quote(xpath))
    elif xpath.startswith('%') and xpath.endswith('%'):
        xpath = '//*[contains(@text, {0}) or contains(@content-desc, {0})]'.format(string_quote(xpath[1:-1]))
    elif xpath.startswith('%'):
        text = xpath[1:]
        xpath = '//*[{0} = substring(@text, string-length(@text) - {1} + 1) or {0} = substring(@content-desc, string-length(@text) - {1} + 1)]'.format(string_quote(text), len(text))
    elif xpath.endswith('%'):
        text = xpath[:-1]
        xpath = '//*[starts-with(@text, {0}) or starts-with(@content-desc, {0})]'.format(string_quote(text))
    else:
        xpath = '//*[@text={0} or @content-desc={0} or @resource-id={0}]'.format(string_quote(xpath))
    logger.debug('xpath %s -> %s', orig_xpath, xpath)
    return xpath

class TimeoutException(Exception):
    pass

class XPathError(Exception):
    """ basic error for xpath plugin """

class UIMeta(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def click(self, x: int, y: int):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abc.abstractmethod
    def swipe(self, fx: int, fy: int, tx: int, ty: int, duration: float):
        if False:
            while True:
                i = 10
        ' duration is float type, indicate seconds '

    @abc.abstractmethod
    def window_size(self) -> tuple:
        if False:
            return 10
        ' return (width, height) '

    @abc.abstractmethod
    def dump_hierarchy(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        ' return xml content '

    @abc.abstractmethod
    def screenshot(self):
        if False:
            return 10
        ' return PIL.Image.Image '

class XPath(object):

    def __init__(self, d: 'uiautomator2.Device'):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            d (uiautomator2 instance)\n        '
        self._d = d
        assert hasattr(d, 'click')
        assert hasattr(d, 'swipe')
        assert hasattr(d, 'window_size')
        assert hasattr(d, 'dump_hierarchy')
        assert hasattr(d, 'screenshot')
        assert hasattr(d, 'wait_timeout')
        self._click_before_delay = 0.0
        self._click_after_delay = None
        self._last_source = None
        self._event_callbacks = defaultdict(list)
        self._alias = {}
        self._alias_strict = False
        self._dump_lock = threading.Lock()
        self._logger = setup_logger()
        self._logger.setLevel(logging.INFO)

    def global_set(self, key, value):
        if False:
            while True:
                i = 10
        valid_keys = {'timeout', 'alias', 'alias_strict', 'click_after_delay', 'click_before_delay'}
        if key not in valid_keys:
            raise ValueError('invalid key', key)
        if key == 'timeout':
            self.implicitly_wait(value)
        else:
            setattr(self, '_' + key, value)

    def implicitly_wait(self, timeout):
        if False:
            print('Hello World!')
        ' set default timeout when click '
        self._d.wait_timeout = timeout

    @property
    def logger(self):
        if False:
            print('Hello World!')
        expect_level = logging.DEBUG if self._d.settings['xpath_debug'] else logging.INFO
        if expect_level != self._logger.level:
            self._logger.setLevel(expect_level)
        return self._logger

    @property
    def wait_timeout(self):
        if False:
            return 10
        return self._d.wait_timeout

    @property
    def _watcher(self):
        if False:
            return 10
        return self._d.watcher

    def dump_hierarchy(self):
        if False:
            i = 10
            return i + 15
        with self._dump_lock:
            self._last_source = self._d.dump_hierarchy()
            return self._last_source

    def get_last_hierarchy(self):
        if False:
            print('Hello World!')
        return self._last_source

    def add_event_listener(self, event_name, callback):
        if False:
            for i in range(10):
                print('nop')
        self._event_callbacks[event_name] += [callback]

    def send_click(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        if self._click_before_delay:
            self.logger.debug('click before delay %.1f seconds', self._click_after_delay)
            time.sleep(self._click_before_delay)
        for callback_func in self._event_callbacks['send_click']:
            callback_func(x, y)
        self._d.click(x, y)
        if self._click_after_delay:
            self.logger.debug('click after delay %.1f seconds', self._click_after_delay)
            time.sleep(self._click_after_delay)

    def send_longclick(self, x, y):
        if False:
            return 10
        self._d.long_click(x, y)

    def send_swipe(self, sx, sy, tx, ty):
        if False:
            while True:
                i = 10
        self._d.swipe(sx, sy, tx, ty)

    def send_text(self, text: str=None):
        if False:
            while True:
                i = 10
        self._d.set_fastinput_ime()
        self._d.clear_text()
        if text:
            self._d.send_keys(text)

    def take_screenshot(self) -> Image.Image:
        if False:
            print('Hello World!')
        return self._d.screenshot()

    def match(self, xpath, source=None):
        if False:
            i = 10
            return i + 15
        return len(self(xpath, source).all()) > 0

    @deprecated(version='3.0.0', reason='use d.watcher.when(..) instead')
    def when(self, xquery: str):
        if False:
            while True:
                i = 10
        return self._watcher.when(xquery)

    @deprecated(version='3.0.0', reason='deprecated')
    def apply_watch_from_yaml(self, data):
        if False:
            while True:
                i = 10
        '\n        Examples of argument data\n\n            ---\n            - when: "@com.example.app/popup"\n            then: >\n                def callback(d):\n                    d.click(0.5, 0.5)\n            - when: 继续\n            then: click\n        '
        try:
            import yaml
        except ImportError:
            self.logger.warning('missing lib pyyaml')
        data = yaml.load(data, Loader=yaml.SafeLoader)
        for item in data:
            (when, then) = (item['when'], item['then'])
            trigger = lambda : None
            self.logger.info('%s, %s', when, then)
            if then == 'click':
                trigger = lambda selector: selector.get_last_match().click()
                trigger.__doc__ = 'click'
            elif then.lstrip().startswith('def callback'):
                mod = ModuleType('_inner_module')
                exec(then, mod.__dict__)
                trigger = mod.callback
                trigger.__doc__ = then
            else:
                self.logger.warning('Unknown then: %r', then)
            self.logger.debug('When: %r, Trigger: %r', when, trigger.__doc__)
            self.when(when).call(trigger)

    @deprecated(version='3.0.0', reason='use d.watcher.run() instead')
    def run_watchers(self, source=None):
        if False:
            while True:
                i = 10
        self._watcher.run()

    @deprecated(version='3.0.0', reason='use d.watcher.start(..) instead')
    def watch_background(self, interval: float=4.0):
        if False:
            for i in range(10):
                print('nop')
        return self._watcher.start(interval)

    @deprecated(version='3.0.0', reason='use d.watcher.stop() instead')
    def watch_stop(self):
        if False:
            for i in range(10):
                print('nop')
        ' stop watch background '
        self._watcher.stop()

    @deprecated(version='3.0.0', reason='use d.watcher.remove() instead')
    def watch_clear(self):
        if False:
            for i in range(10):
                print('nop')
        self._watcher.stop()

    @deprecated(version='3.0.0', reason='removed')
    def sleep_watch(self, seconds):
        if False:
            return 10
        ' run watchers when sleep '
        deadline = time.time() + seconds
        while time.time() < deadline:
            self.run_watchers()
            left_time = max(0, deadline - time.time())
            time.sleep(min(0.5, left_time))

    def _get_after_watch(self, xpath: Union[str, list], timeout=None):
        if False:
            return 10
        if timeout == 0:
            timeout = 0.01
        timeout = timeout or self.wait_timeout
        self.logger.info('XPath(timeout %.1f) %s', timeout, xpath)
        deadline = time.time() + timeout
        while True:
            source = self.dump_hierarchy()
            selector = self(xpath, source)
            if selector.exists:
                return selector.get_last_match()
            if time.time() > deadline:
                break
            time.sleep(0.5)
        raise TimeoutException('timeout %.1f, xpath: %s' % (timeout, xpath))

    def click(self, xpath: Union[str, list], timeout=None, pre_delay: float=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find element and perform click\n\n        Args:\n            xpath (str): xpath string\n            timeout (float): pass\n            pre_delay (float): pre delay wait time before click\n\n        Raises:\n            TimeoutException\n        '
        el = self._get_after_watch(xpath, timeout)
        el.click()

    def scroll_to(self, xpath: str, direction: Union[Direction, str]=Direction.FORWARD, max_swipes=10) -> Union['XMLElement', None]:
        if False:
            print('Hello World!')
        '\n        Need more tests\n        scroll up the whole screen until target element founded\n\n        Returns:\n            bool (found or not)\n        '
        if direction == Direction.FORWARD:
            direction = Direction.UP
        elif direction == Direction.BACKWARD:
            direction = Direction.DOWN
        elif direction == Direction.HORIZ_FORWARD:
            direction = Direction.LEFT
        elif direction == Direction.HBACKWORD:
            direction = Direction.RIGHT
        assert max_swipes > 0
        target = self(xpath)
        for i in range(max_swipes):
            if target.exists:
                self._d.swipe_ext(direction, 0.1)
                return target.get_last_match()
            self._d.swipe_ext(direction, 0.5)
        return False

    def __alias_get(self, key, default=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        when alias_strict set, if key not in _alias, XPathError will be raised\n        '
        value = self._alias.get(key, default)
        if value is None:
            if self._alias_strict:
                raise XPathError('alias have not found key', key)
            value = key
        return value

    def __call__(self, xpath: str, source=None):
        if False:
            i = 10
            return i + 15
        return XPathSelector(self, xpath, source)

class XPathSelector(object):

    def __init__(self, parent: XPath, xpath: Union[list, str], source=None):
        if False:
            while True:
                i = 10
        self.logger = parent.logger
        self._parent = parent
        self._d = parent._d
        self._source = source
        self._last_source = None
        self._position = None
        self._fallback = None
        self._xpath_list = []
        self.xpath(xpath)

    def __str__(self):
        if False:
            print('Hello World!')
        return f"XPathSelector({'|'.join(self._xpath_list)}"

    def xpath(self, _xpath: Union[list, tuple, str]):
        if False:
            return 10
        if isinstance(_xpath, str):
            _xpath = strict_xpath(_xpath, self.logger)
            self._xpath_list.append(_xpath)
        elif isinstance(_xpath, (list, tuple)):
            for xp in _xpath:
                self._xpath_list.append(strict_xpath(xp, self.logger))
        else:
            raise TypeError('Unknown type for value {}'.format(_xpath))
        return self

    def child(self, _xpath: str):
        if False:
            i = 10
            return i + 15
        if not _xpath.startswith('/'):
            _xpath = '/' + _xpath
        self._xpath_list[-1] = self._xpath_list[-1] + _xpath
        return self

    def position(self, x: float, y: float):
        if False:
            return 10
        ' set possible position '
        assert 0 < x < 1
        assert 0 < y < 1
        self._position = (x, y)
        return self

    def fallback(self, func: Optional[Callable[..., bool]]=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        callback on failure\n        '
        if isinstance(func, str):
            if func == 'click':
                if len(args) == 0:
                    args = self._position
                func = lambda d: d.click(*args)
            else:
                raise ValueError('func should be "click" or callable function')
        assert callable(func)
        self._fallback = func
        return self

    @property
    def _global_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        return self._parent.wait_timeout

    def all(self, source=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n            list of XMLElement\n        '
        xml_content = source or self._source or self._parent.dump_hierarchy()
        self._last_source = xml_content
        hierarchy = source or self._source
        if not hierarchy:
            trigger_count = 0
            for _ in range(5):
                triggered = self._parent._watcher.run(xml_content)
                if not triggered:
                    break
                trigger_count += 1
                xml_content = self._parent.dump_hierarchy()
            if trigger_count:
                self.logger.debug('watcher triggered %d times', trigger_count)
        if hierarchy is None:
            root = etree.fromstring(str2bytes(xml_content))
        elif isinstance(hierarchy, (str, bytes)):
            root = etree.fromstring(str2bytes(hierarchy))
        elif isinstance(hierarchy, etree._Element):
            root = hierarchy
        else:
            raise TypeError('Unknown type', type(hierarchy))
        for node in root.xpath('//node'):
            node.tag = safe_xmlstr(node.attrib.pop('class', '')) or 'node'
        match_sets = []
        for xpath in self._xpath_list:
            matches = root.xpath(xpath, namespaces={'re': 'http://exslt.org/regular-expressions'})
            match_sets.append(matches)
        match_nodes = functools.reduce(lambda x, y: set(x).intersection(y), match_sets)
        els = [XMLElement(node, self._parent) for node in match_nodes]
        if not self._position:
            return els
        inside_els = []
        (px, py) = self._position
        wsize = self._d.window_size()
        for e in els:
            (lpx, lpy, rpx, rpy) = e.percent_bounds(wsize=wsize)
            scale = 1.5
            if abs(px - (lpx + rpx) / 2) > (rpx - lpx) * 0.5 * scale:
                continue
            if abs(py - (lpy + rpy) / 2) > (rpy - lpy) * 0.5 * scale:
                continue
            inside_els.append(e)
        return inside_els

    @property
    def exists(self):
        if False:
            i = 10
            return i + 15
        return len(self.all()) > 0

    def get(self, timeout=None):
        if False:
            i = 10
            return i + 15
        '\n        Get first matched element\n\n        Args:\n            timeout (float): max seconds to wait\n\n        Returns:\n            XMLElement\n\n        Raises:\n            XPathElementNotFoundError\n        '
        if not self.wait(timeout or self._global_timeout):
            raise XPathElementNotFoundError(self._xpath_list)
        return self.get_last_match()

    def get_last_match(self):
        if False:
            while True:
                i = 10
        return self.all(self._last_source)[0]

    def get_text(self):
        if False:
            return 10
        '\n        get element text\n\n        Returns:\n            string of node text\n\n        Raises:\n            XPathElementNotFoundError\n        '
        return self.get().attrib.get('text', '')

    def set_text(self, text: str=''):
        if False:
            i = 10
            return i + 15
        el = self.get()
        self._d.set_fastinput_ime()
        el.click()
        self._parent.send_text(text)

    def wait(self, timeout=None) -> Optional['XMLElement']:
        if False:
            print('Hello World!')
        '\n        Args:\n            timeout (float): seconds\n\n        Returns:\n            None or XMLElement\n        '
        deadline = time.time() + (timeout or self._global_timeout)
        while time.time() < deadline:
            if self.exists:
                return self.get_last_match()
            time.sleep(0.2)
        return None

    def match(self) -> Optional['XMLElement']:
        if False:
            print('Hello World!')
        '\n        Returns:\n            None or matched XMLElement\n        '
        if self.exists:
            return self.get_last_match()

    def wait_gone(self, timeout=None) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            timeout (float): seconds\n\n        Returns:\n            True if gone else False\n        '
        deadline = time.time() + (timeout or self._global_timeout)
        while time.time() < deadline:
            if not self.exists:
                return True
            time.sleep(0.2)
        return False

    def click_nowait(self):
        if False:
            i = 10
            return i + 15
        (x, y) = self.all()[0].center()
        self.logger.info('click %d, %d', x, y)
        self._parent.send_click(x, y)

    def click(self, timeout=None):
        if False:
            i = 10
            return i + 15
        ' find element and perform click '
        try:
            el = self.get(timeout=timeout)
            el.click()
        except XPathElementNotFoundError:
            if not self._fallback:
                raise
            self.logger.info('element not found, run fallback')
            return inject_call(self._fallback, d=self._d)

    def click_exists(self, timeout=None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        el = self.wait(timeout=timeout)
        if el:
            el.click()
            return True
        return False

    def long_click(self):
        if False:
            for i in range(10):
                print('nop')
        ' find element and perform long click '
        self.get().long_click()

    def screenshot(self) -> Image.Image:
        if False:
            for i in range(10):
                print('nop')
        ' take element screenshot '
        el = self.get()
        return el.screenshot()

    def __getattr__(self, key: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        In IPython console, attr:_ipython_canary_method_should_not_exist_ will be called\n        So here ignore all attr startswith _\n        '
        if key.startswith('_'):
            raise AttributeError('Invalid attr', key)
        el = self.get()
        return getattr(el, key)

class XMLElement(object):

    def __init__(self, elem, parent: XPath):
        if False:
            return 10
        '\n        Args:\n            elem: lxml node\n            d: uiautomator2 instance\n        '
        self.elem = elem
        self._parent = parent
        self._d = parent._d

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        compared_attrs = ('text', 'resource-id', 'package', 'content-desc')
        values = [self.attrib.get(name) for name in compared_attrs]
        root = self.elem.getroottree()
        fullpath = root.getpath(self.elem)
        fullpath = re.sub('\\[\\d+\\]', '', fullpath)
        values.append(fullpath)
        return hash(tuple(values))

    def __eq__(self, value):
        if False:
            return 10
        return self.__hash__() == hash(value)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        (x, y) = self.center()
        return '<xpath.XMLElement [{tag!r} center:({x}, {y})]>'.format(tag=self.elem.tag, x=x, y=y)

    def get_xpath(self, strip_index: bool=False):
        if False:
            i = 10
            return i + 15
        ' get element full xpath '
        root = self.elem.getroottree()
        path = root.getpath(self.elem)
        if strip_index:
            path = re.sub('\\[\\d+\\]', '', path)
        return path

    def center(self):
        if False:
            print('Hello World!')
        '\n        Returns:\n            (x, y)\n        '
        return self.offset(0.5, 0.5)

    def offset(self, px: float=0.0, py: float=0.0):
        if False:
            print('Hello World!')
        '\n        Offset from left_top\n\n        Args:\n            px (float): percent of width\n            py (float): percent of height\n\n        Example:\n            offset(0.5, 0.5) means center\n        '
        (x, y, width, height) = self.rect
        return (x + int(width * px), y + int(height * py))

    def click(self):
        if False:
            return 10
        '\n        click element, 100ms between down and up\n        '
        (x, y) = self.center()
        self._parent.send_click(x, y)

    def long_click(self):
        if False:
            print('Hello World!')
        '\n        Sometime long click is needed, 400ms between down and up\n        '
        (x, y) = self.center()
        self._parent.send_longclick(x, y)

    def screenshot(self):
        if False:
            print('Hello World!')
        '\n        Take screenshot of element\n        '
        im = self._parent.take_screenshot()
        return im.crop(self.bounds)

    def swipe(self, direction: Union[Direction, str], scale: float=0.6):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            direction: one of ["left", "right", "up", "down"]\n            scale: percent of swipe, range (0, 1.0)\n        \n        Raises:\n            AssertionError, ValueError\n        '
        return swipe_in_bounds(self._parent._d, self.bounds, direction, scale)

    def scroll(self, direction: Union[Direction, str]=Direction.FORWARD) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            direction: Direction eg: Direction.FORWARD\n        \n        Returns:\n            bool: if can be scroll again\n        '
        if direction == 'forward':
            direction = Direction.FORWARD
        elif direction == 'backward':
            direction = Direction.BACKWARD
        els = set(self._parent('//*').all())
        self.swipe(direction, scale=0.6)
        new_elements = set(self._parent('//*').all()) - els
        ppath = self.get_xpath() + '/'
        els = [el for el in new_elements if el.get_xpath().startswith(ppath)]
        return len(els) > 0

    def scroll_to(self, xpath: str, direction: Direction=Direction.FORWARD, max_swipes: int=10) -> Union['XMLElement', None]:
        if False:
            return 10
        assert max_swipes > 0
        target = self._parent(xpath)
        for i in range(max_swipes):
            if target.exists:
                return target.get_last_match()
            if not self.scroll(direction):
                break
        return None

    def parent(self, xpath: Optional[str]=None) -> Union['XMLElement', None]:
        if False:
            print('Hello World!')
        '\n        Returns parent element\n        '
        if xpath is None:
            return XMLElement(self.elem.getparent(), self._parent)
        root = self.elem.getroottree()
        e = self.elem
        els = []
        while e is not None and e != root:
            els.append(e)
            e = e.getparent()
        xpath = strict_xpath(xpath)
        matches = root.xpath(xpath, namespaces={'re': 'http://exslt.org/regular-expressions'})
        all_paths = [root.getpath(m) for m in matches]
        for e in reversed(els):
            if root.getpath(e) in all_paths:
                return XMLElement(e, self._parent)

    def percent_size(self):
        if False:
            while True:
                i = 10
        ' Returns:\n                (float, float): eg, (0.5, 0.5) means 50%, 50%\n        '
        (ww, wh) = self._d.window_size()
        (_, _, w, h) = self.rect
        return (w / ww, h / wh)

    @property
    def bounds(self):
        if False:
            return 10
        '\n        Returns:\n            tuple of (left, top, right, bottom)\n        '
        bounds = self.elem.attrib.get('bounds')
        (lx, ly, rx, ry) = map(int, re.findall('\\d+', bounds))
        return (lx, ly, rx, ry)

    def percent_bounds(self, wsize: Optional[tuple]=None):
        if False:
            while True:
                i = 10
        ' \n        Args:\n            wsize (tuple(int, int)): window size\n        \n        Returns:\n            list of 4 float, eg: 0.1, 0.2, 0.5, 0.8\n        '
        (lx, ly, rx, ry) = self.bounds
        (ww, wh) = wsize or self._d.window_size()
        return (lx / ww, ly / wh, rx / ww, ry / wh)

    @property
    def rect(self):
        if False:
            return 10
        '\n        Returns:\n            (left_top_x, left_top_y, width, height)\n        '
        (lx, ly, rx, ry) = self.bounds
        return (lx, ly, rx - lx, ry - ly)

    @property
    def text(self):
        if False:
            for i in range(10):
                print('nop')
        return self.elem.attrib.get('text')

    @property
    def attrib(self):
        if False:
            i = 10
            return i + 15
        return self.elem.attrib

    @property
    def info(self):
        if False:
            while True:
                i = 10
        ret = {}
        for key in ('text', 'focusable', 'enabled', 'focused', 'scrollable', 'selected', 'clickable'):
            ret[key] = self.attrib.get(key)
        ret['className'] = self.elem.tag
        (lx, ly, rx, ry) = self.bounds
        ret['bounds'] = {'left': lx, 'top': ly, 'right': rx, 'bottom': ry}
        ret['contentDescription'] = self.attrib.get('content-desc')
        ret['longClickable'] = self.attrib.get('long-clickable')
        ret['packageName'] = self.attrib.get('package')
        ret['resourceName'] = self.attrib.get('resource-id')
        ret['resourceId'] = self.attrib.get('resource-id')
        ret['childCount'] = len(self.elem.getchildren())
        return ret

class AdbUI(BasicUIMeta):
    """
    Use adb command to run ui test
    """

    def __init__(self, d: adbutils.AdbDevice):
        if False:
            while True:
                i = 10
        self._d = d

    def click(self, x, y):
        if False:
            i = 10
            return i + 15
        self._d.click(x, y)

    def swipe(self, sx, sy, ex, ey, duration):
        if False:
            i = 10
            return i + 15
        self._d.swipe(sx, sy, ex, ey, duration)

    def window_size(self):
        if False:
            return 10
        (w, h) = self._d.window_size()
        return (w, h)

    def dump_hierarchy(self):
        if False:
            print('Hello World!')
        return self._d.dump_hierarchy()

    def screenshot(self):
        if False:
            i = 10
            return i + 15
        d = self._d
        json_output = d.shell(['LD_LIBRARY_PATH=/data/local/tmp', '/data/local/tmp/minicap', '-i', '2&>/dev/null']).strip()
        data = json.loads(json_output)
        (w, h, r) = (data['width'], data['height'], data['rotation'])
        remote_image_path = '/sdcard/minicap.jpg'
        d.shell(['rm', remote_image_path])
        d.shell(['LD_LIBRARY_PATH=/data/local/tmp', '/data/local/tmp/minicap', '-P', '{0}x{1}@{0}x{1}/{2}'.format(w, h, r), '-s', '>' + remote_image_path])
        if d.sync.stat(remote_image_path).size == 0:
            raise RuntimeError('screenshot using minicap error')
        buf = io.BytesIO()
        for data in d.sync.iter_content(remote_image_path):
            buf.write(data)
        return Image.open(buf)
if __name__ == '__main__':
    d = AdbUI(adbutils.adb.device())
    xpath = XPath(d)
    xpath('App').click()
    xpath('Alarm').click()