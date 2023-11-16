import logging
import time
import warnings
import requests
import six
from PIL import Image
from retry import retry
from .exceptions import UiObjectNotFoundError
from .utils import Exists, intersect
from ._proto import SCROLL_STEPS

class Selector(dict):
    """The class is to build parameters for UiSelector passed to Android device.
    """
    __fields = {'text': (1, None), 'textContains': (2, None), 'textMatches': (4, None), 'textStartsWith': (8, None), 'className': (16, None), 'classNameMatches': (32, None), 'description': (64, None), 'descriptionContains': (128, None), 'descriptionMatches': (256, None), 'descriptionStartsWith': (512, None), 'checkable': (1024, False), 'checked': (2048, False), 'clickable': (4096, False), 'longClickable': (8192, False), 'scrollable': (16384, False), 'enabled': (32768, False), 'focusable': (65536, False), 'focused': (131072, False), 'selected': (262144, False), 'packageName': (524288, None), 'packageNameMatches': (1048576, None), 'resourceId': (2097152, None), 'resourceIdMatches': (4194304, None), 'index': (8388608, 0), 'instance': (16777216, 0)}
    (__mask, __childOrSibling, __childOrSiblingSelector) = ('mask', 'childOrSibling', 'childOrSiblingSelector')

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Selector, self).__setitem__(self.__mask, 0)
        super(Selector, self).__setitem__(self.__childOrSibling, [])
        super(Selector, self).__setitem__(self.__childOrSiblingSelector, [])
        for k in kwargs:
            self[k] = kwargs[k]

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        ' remove useless part for easily debugger '
        selector = self.copy()
        selector.pop('mask')
        for key in ('childOrSibling', 'childOrSiblingSelector'):
            if not selector.get(key):
                selector.pop(key)
        args = []
        for (k, v) in selector.items():
            args.append(k + '=' + repr(v))
        return 'Selector [' + ', '.join(args) + ']'

    def __setitem__(self, k, v):
        if False:
            return 10
        if k in self.__fields:
            super(Selector, self).__setitem__(k, v)
            super(Selector, self).__setitem__(self.__mask, self[self.__mask] | self.__fields[k][0])
        else:
            raise ReferenceError('%s is not allowed.' % k)

    def __delitem__(self, k):
        if False:
            return 10
        if k in self.__fields:
            super(Selector, self).__delitem__(k)
            super(Selector, self).__setitem__(self.__mask, self[self.__mask] & ~self.__fields[k][0])

    def clone(self):
        if False:
            return 10
        kwargs = dict(((k, self[k]) for k in self if k not in [self.__mask, self.__childOrSibling, self.__childOrSiblingSelector]))
        selector = Selector(**kwargs)
        for v in self[self.__childOrSibling]:
            selector[self.__childOrSibling].append(v)
        for s in self[self.__childOrSiblingSelector]:
            selector[self.__childOrSiblingSelector].append(s.clone())
        return selector

    def child(self, **kwargs):
        if False:
            return 10
        self[self.__childOrSibling].append('child')
        self[self.__childOrSiblingSelector].append(Selector(**kwargs))
        return self

    def sibling(self, **kwargs):
        if False:
            return 10
        self[self.__childOrSibling].append('sibling')
        self[self.__childOrSiblingSelector].append(Selector(**kwargs))
        return self

    def update_instance(self, i):
        if False:
            return 10
        if self[self.__childOrSiblingSelector]:
            self[self.__childOrSiblingSelector][-1]['instance'] = i
        else:
            self['instance'] = i

class UiObject(object):

    def __init__(self, session, selector: Selector):
        if False:
            print('Hello World!')
        self.session = session
        self.selector = selector
        self.jsonrpc = session.jsonrpc

    @property
    def wait_timeout(self):
        if False:
            i = 10
            return i + 15
        return self.session.wait_timeout

    @property
    def exists(self):
        if False:
            while True:
                i = 10
        'check if the object exists in current window.'
        return Exists(self)

    @property
    @retry(UiObjectNotFoundError, delay=0.5, tries=3, jitter=0.1, logger=logging)
    def info(self):
        if False:
            for i in range(10):
                print('nop')
        'ui object info.'
        return self.jsonrpc.objInfo(self.selector)

    def screenshot(self) -> Image.Image:
        if False:
            i = 10
            return i + 15
        im = self.session.screenshot()
        return im.crop(self.bounds())

    def click(self, timeout=None, offset=None):
        if False:
            print('Hello World!')
        '\n        Click UI element. \n\n        Args:\n            timeout: seconds wait element show up\n            offset: (xoff, yoff) default (0.5, 0.5) -> center\n\n        The click method does the same logic as java uiautomator does.\n        1. waitForExists 2. get VisibleBounds center 3. send click event\n\n        Raises:\n            UiObjectNotFoundError\n        '
        self.must_wait(timeout=timeout)
        (x, y) = self.center(offset=offset)
        self.session.click(x, y)

    def bounds(self):
        if False:
            while True:
                i = 10
        '\n        Returns:\n            left_top_x, left_top_y, right_bottom_x, right_bottom_y\n        '
        info = self.info
        bounds = info.get('visibleBounds') or info.get('bounds')
        (lx, ly, rx, ry) = (bounds['left'], bounds['top'], bounds['right'], bounds['bottom'])
        return (lx, ly, rx, ry)

    def center(self, offset=(0.5, 0.5)):
        if False:
            return 10
        '\n        Args:\n            offset: optional, (x_off, y_off)\n                (0, 0) means left-top, (0.5, 0.5) means middle(Default)\n        Return:\n            center point (x, y)\n        '
        (lx, ly, rx, ry) = self.bounds()
        if offset is None:
            offset = (0.5, 0.5)
        (xoff, yoff) = offset
        (width, height) = (rx - lx, ry - ly)
        x = lx + width * xoff
        y = ly + height * yoff
        return (x, y)

    def click_gone(self, maxretry=10, interval=1.0):
        if False:
            print('Hello World!')
        '\n        Click until element is gone\n\n        Args:\n            maxretry (int): max click times\n            interval (float): sleep time between clicks\n\n        Return:\n            Bool if element is gone\n        '
        self.click_exists()
        while maxretry > 0:
            time.sleep(interval)
            if not self.exists:
                return True
            self.click_exists()
            maxretry -= 1
        return False

    def click_exists(self, timeout=0):
        if False:
            while True:
                i = 10
        try:
            self.click(timeout=timeout)
            return True
        except UiObjectNotFoundError:
            return False

    def long_click(self, duration: float=0.5, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            duration (float): seconds of pressed\n            timeout (float): seconds wait element show up\n        '
        self.must_wait(timeout=timeout)
        (x, y) = self.center()
        return self.session.long_click(x, y, duration)

    def drag_to(self, *args, **kwargs):
        if False:
            return 10
        duration = kwargs.pop('duration', 0.5)
        timeout = kwargs.pop('timeout', None)
        self.must_wait(timeout=timeout)
        steps = int(duration * 200)
        if len(args) >= 2 or 'x' in kwargs or 'y' in kwargs:

            def drag2xy(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                (x, y) = self.session.pos_rel2abs(x, y)
                return self.jsonrpc.dragTo(self.selector, x, y, steps)
            return drag2xy(*args, **kwargs)
        return self.jsonrpc.dragTo(self.selector, Selector(**kwargs), steps)

    def swipe(self, direction, steps=10):
        if False:
            while True:
                i = 10
        '\n        Performs the swipe action on the UiObject.\n        Swipe from center\n\n        Args:\n            direction (str): one of ("left", "right", "up", "down")\n            steps (int): move steps, one step is about 5ms\n            percent: float between [0, 1]\n\n        Note: percent require API >= 18\n        # assert 0 <= percent <= 1\n        '
        assert direction in ('left', 'right', 'up', 'down')
        self.must_wait()
        info = self.info
        bounds = info.get('visibleBounds') or info.get('bounds')
        (lx, ly, rx, ry) = (bounds['left'], bounds['top'], bounds['right'], bounds['bottom'])
        (cx, cy) = ((lx + rx) // 2, (ly + ry) // 2)
        if direction == 'up':
            self.session.swipe(cx, cy, cx, ly, steps=steps)
        elif direction == 'down':
            self.session.swipe(cx, cy, cx, ry - 1, steps=steps)
        elif direction == 'left':
            self.session.swipe(cx, cy, lx, cy, steps=steps)
        elif direction == 'right':
            self.session.swipe(cx, cy, rx - 1, cy, steps=steps)

    def gesture(self, start1, start2, end1, end2, steps=100):
        if False:
            for i in range(10):
                print('nop')
        '\n        perform two point gesture.\n        Usage:\n        d().gesture(startPoint1, startPoint2, endPoint1, endPoint2, steps)\n        '
        rel2abs = self.session.pos_rel2abs

        def point(x=0, y=0):
            if False:
                while True:
                    i = 10
            (x, y) = rel2abs(x, y)
            return {'x': x, 'y': y}

        def ctp(pt):
            if False:
                print('Hello World!')
            return point(*pt) if type(pt) == tuple else pt
        (s1, s2, e1, e2) = (ctp(start1), ctp(start2), ctp(end1), ctp(end2))
        return self.jsonrpc.gesture(self.selector, s1, s2, e1, e2, steps)

    def pinch_in(self, percent=100, steps=50):
        if False:
            i = 10
            return i + 15
        return self.jsonrpc.pinchIn(self.selector, percent, steps)

    def pinch_out(self, percent=100, steps=50):
        if False:
            return 10
        return self.jsonrpc.pinchOut(self.selector, percent, steps)

    def wait(self, exists=True, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wait until UI Element exists or gone\n\n        Args:\n            timeout (float): wait element timeout\n\n        Example:\n            d(text="Clock").wait()\n            d(text="Settings").wait(exists=False) # wait until it\'s gone\n        '
        if timeout is None:
            timeout = self.wait_timeout
        http_wait = timeout + 10
        if exists:
            try:
                return self.jsonrpc.waitForExists(self.selector, int(timeout * 1000), http_timeout=http_wait)
            except requests.ReadTimeout as e:
                warnings.warn('waitForExists readTimeout: %s' % e, RuntimeWarning)
                return self.exists()
        else:
            try:
                return self.jsonrpc.waitUntilGone(self.selector, int(timeout * 1000), http_timeout=http_wait)
            except requests.ReadTimeout as e:
                warnings.warn('waitForExists readTimeout: %s' % e, RuntimeWarning)
                return not self.exists()

    def wait_gone(self, timeout=None):
        if False:
            return 10
        ' wait until ui gone\n        Args:\n            timeout (float): wait element gone timeout\n\n        Returns:\n            bool if element gone\n        '
        timeout = timeout or self.wait_timeout
        return self.wait(exists=False, timeout=timeout)

    def must_wait(self, exists=True, timeout=None):
        if False:
            i = 10
            return i + 15
        ' wait and if not found raise UiObjectNotFoundError '
        if not self.wait(exists, timeout):
            raise UiObjectNotFoundError({'code': -32002, 'data': str(self.selector), 'method': 'wait'})

    def send_keys(self, text):
        if False:
            print('Hello World!')
        ' alias of set_text '
        return self.set_text(text)

    def set_text(self, text, timeout=None):
        if False:
            return 10
        self.must_wait(timeout=timeout)
        if not text:
            return self.jsonrpc.clearTextField(self.selector)
        else:
            return self.jsonrpc.setText(self.selector, text)

    def get_text(self, timeout=None):
        if False:
            return 10
        ' get text from field '
        self.must_wait(timeout=timeout)
        return self.jsonrpc.getText(self.selector)

    def clear_text(self, timeout=None):
        if False:
            while True:
                i = 10
        self.must_wait(timeout=timeout)
        return self.set_text(None)

    def child(self, **kwargs):
        if False:
            while True:
                i = 10
        return UiObject(self.session, self.selector.clone().child(**kwargs))

    def sibling(self, **kwargs):
        if False:
            print('Hello World!')
        return UiObject(self.session, self.selector.clone().sibling(**kwargs))
    (child_selector, from_parent) = (child, sibling)

    def child_by_text(self, txt, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'allow_scroll_search' in kwargs:
            allow_scroll_search = kwargs.pop('allow_scroll_search')
            name = self.jsonrpc.childByText(self.selector, Selector(**kwargs), txt, allow_scroll_search)
        else:
            name = self.jsonrpc.childByText(self.selector, Selector(**kwargs), txt)
        return UiObject(self.session, name)

    def child_by_description(self, txt, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'allow_scroll_search' in kwargs:
            allow_scroll_search = kwargs.pop('allow_scroll_search')
            name = self.jsonrpc.childByDescription(self.selector, Selector(**kwargs), txt, allow_scroll_search)
        else:
            name = self.jsonrpc.childByDescription(self.selector, Selector(**kwargs), txt)
        return UiObject(self.session, name)

    def child_by_instance(self, inst, **kwargs):
        if False:
            while True:
                i = 10
        return UiObject(self.session, self.jsonrpc.childByInstance(self.selector, Selector(**kwargs), inst))

    def parent(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def __getitem__(self, instance: int):
        if False:
            while True:
                i = 10
        '\n        Raises:\n            IndexError\n        '
        if isinstance(self.selector, six.string_types):
            raise IndexError('Index is not supported when UiObject returned by child_by_xxx')
        selector = self.selector.clone()
        if instance < 0:
            selector['instance'] = 0
            del selector['instance']
            count = self.jsonrpc.count(selector)
            assert instance + count >= 0
            instance += count
        selector.update_instance(instance)
        return UiObject(self.session, selector)

    @property
    def count(self):
        if False:
            i = 10
            return i + 15
        return self.jsonrpc.count(self.selector)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.count

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        (obj, length) = (self, self.count)

        class Iter(object):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.index = -1

            def next(self):
                if False:
                    while True:
                        i = 10
                self.index += 1
                if self.index < length:
                    return obj[self.index]
                else:
                    raise StopIteration()
            __next__ = next
        return Iter()

    def right(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')

        def onrightof(rect1, rect2):
            if False:
                for i in range(10):
                    print('nop')
            (left, top, right, bottom) = intersect(rect1, rect2)
            return rect2['left'] - rect1['right'] if top < bottom else -1
        return self.__view_beside(onrightof, **kwargs)

    def left(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')

        def onleftof(rect1, rect2):
            if False:
                for i in range(10):
                    print('nop')
            (left, top, right, bottom) = intersect(rect1, rect2)
            return rect1['left'] - rect2['right'] if top < bottom else -1
        return self.__view_beside(onleftof, **kwargs)

    def up(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')

        def above(rect1, rect2):
            if False:
                return 10
            (left, top, right, bottom) = intersect(rect1, rect2)
            return rect1['top'] - rect2['bottom'] if left < right else -1
        return self.__view_beside(above, **kwargs)

    def down(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')

        def under(rect1, rect2):
            if False:
                i = 10
                return i + 15
            (left, top, right, bottom) = intersect(rect1, rect2)
            return rect2['top'] - rect1['bottom'] if left < right else -1
        return self.__view_beside(under, **kwargs)

    def __view_beside(self, onsideof, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        bounds = self.info['bounds']
        (min_dist, found) = (-1, None)
        for ui in UiObject(self.session, Selector(**kwargs)):
            dist = onsideof(bounds, ui.info['bounds'])
            if dist >= 0 and (min_dist < 0 or dist < min_dist):
                (min_dist, found) = (dist, ui)
        return found

    @property
    def fling(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            dimention (str): one of "vert", "vertically", "vertical", "horiz", "horizental", "horizentally"\n            action (str): one of "forward", "backward", "toBeginning", "toEnd", "to"\n        '
        jsonrpc = self.jsonrpc
        selector = self.selector

        class _Fling(object):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.vertical = True
                self.action = 'forward'

            def __getattr__(self, key):
                if False:
                    while True:
                        i = 10
                if key in ['horiz', 'horizental', 'horizentally']:
                    self.vertical = False
                    return self
                if key in ['vert', 'vertically', 'vertical']:
                    self.vertical = True
                    return self
                if key in ['forward', 'backward', 'toBeginning', 'toEnd', 'to']:
                    self.action = key
                    return self
                raise ValueError('invalid prop %s' % key)

            def __call__(self, max_swipes=500, **kwargs):
                if False:
                    i = 10
                    return i + 15
                if self.action == 'forward':
                    return jsonrpc.flingForward(selector, self.vertical)
                elif self.action == 'backward':
                    return jsonrpc.flingBackward(selector, self.vertical)
                elif self.action == 'toBeginning':
                    return jsonrpc.flingToBeginning(selector, self.vertical, max_swipes)
                elif self.action == 'toEnd':
                    return jsonrpc.flingToEnd(selector, self.vertical, max_swipes)
        return _Fling()

    @property
    def scroll(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            dimention (str): one of "vert", "vertically", "vertical", "horiz", "horizental", "horizentally"\n            action (str): one of "forward", "backward", "toBeginning", "toEnd", "to"\n        '
        selector = self.selector
        jsonrpc = self.jsonrpc

        class _Scroll(object):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.vertical = True
                self.action = 'forward'

            def __getattr__(self, key):
                if False:
                    i = 10
                    return i + 15
                if key in ['horiz', 'horizental', 'horizentally']:
                    self.vertical = False
                    return self
                if key in ['vert', 'vertically', 'vertical']:
                    self.vertical = True
                    return self
                if key in ['forward', 'backward', 'toBeginning', 'toEnd', 'to']:
                    self.action = key
                    return self
                raise ValueError('invalid prop %s' % key)

            def __call__(self, steps=SCROLL_STEPS, max_swipes=500, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                if self.action in ['forward', 'backward']:
                    method = jsonrpc.scrollForward if self.action == 'forward' else jsonrpc.scrollBackward
                    return method(selector, self.vertical, steps)
                elif self.action == 'toBeginning':
                    return jsonrpc.scrollToBeginning(selector, self.vertical, max_swipes, steps)
                elif self.action == 'toEnd':
                    return jsonrpc.scrollToEnd(selector, self.vertical, max_swipes, steps)
                elif self.action == 'to':
                    return jsonrpc.scrollTo(selector, Selector(**kwargs), self.vertical)
        return _Scroll()