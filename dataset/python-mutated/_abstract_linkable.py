"""
Internal module, support for the linkable protocol for "event" like objects.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from gc import get_objects
from greenlet import greenlet
from greenlet import error as greenlet_error
from gevent._compat import thread_mod_name
from gevent._hub_local import get_hub_noargs as get_hub
from gevent._hub_local import get_hub_if_exists
from gevent.exceptions import InvalidSwitchError
from gevent.exceptions import InvalidThreadUseError
from gevent.timeout import Timeout
locals()['getcurrent'] = __import__('greenlet').getcurrent
locals()['greenlet_init'] = lambda : None
__all__ = ['AbstractLinkable']
_get_thread_ident = __import__(thread_mod_name).get_ident
_allocate_thread_lock = __import__(thread_mod_name).allocate_lock

class _FakeNotifier(object):
    __slots__ = ('pending',)

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.pending = False

def get_roots_and_hubs():
    if False:
        print('Hello World!')
    from gevent.hub import Hub
    return {x.parent: x for x in get_objects() if isinstance(x, Hub) and x.loop is not None}

class AbstractLinkable(object):
    __slots__ = ('hub', '_links', '_notifier', '_notify_all', '__weakref__')

    def __init__(self, hub=None):
        if False:
            while True:
                i = 10
        self._links = []
        self._notifier = None
        self._notify_all = True
        self.hub = hub

    def linkcount(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._links)

    def ready(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def rawlink(self, callback):
        if False:
            print('Hello World!')
        '\n        Register a callback to call when this object is ready.\n\n        *callback* will be called in the :class:`Hub\n        <gevent.hub.Hub>`, so it must not use blocking gevent API.\n        *callback* will be passed one argument: this instance.\n        '
        if not callable(callback):
            raise TypeError('Expected callable: %r' % (callback,))
        self._links.append(callback)
        self._check_and_notify()

    def unlink(self, callback):
        if False:
            i = 10
            return i + 15
        'Remove the callback set by :meth:`rawlink`'
        try:
            self._links.remove(callback)
        except ValueError:
            pass
        if not self._links and self._notifier is not None and self._notifier.pending:
            self._notifier.stop()

    def _allocate_lock(self):
        if False:
            print('Hello World!')
        return _allocate_thread_lock()

    def _getcurrent(self):
        if False:
            while True:
                i = 10
        return getcurrent()

    def _get_thread_ident(self):
        if False:
            print('Hello World!')
        return _get_thread_ident()

    def _capture_hub(self, create):
        if False:
            return 10
        while 1:
            my_hub = self.hub
            if my_hub is None:
                break
            if my_hub.dead:
                if self.hub is my_hub:
                    self.hub = None
                    my_hub = None
                    break
            else:
                break
        if self.hub is None:
            current_hub = get_hub() if create else get_hub_if_exists()
            if self.hub is None:
                self.hub = current_hub
        if self.hub is not None and self.hub.thread_ident != _get_thread_ident():
            raise InvalidThreadUseError(self.hub, get_hub_if_exists(), getcurrent())
        return self.hub

    def _check_and_notify(self):
        if False:
            for i in range(10):
                print('nop')
        if self.ready() and self._links and (not self._notifier):
            hub = None
            try:
                hub = self._capture_hub(False)
            except InvalidThreadUseError:
                pass
            if hub is not None:
                self._notifier = hub.loop.run_callback(self._notify_links, [])
            else:
                self._notifier = 1
                try:
                    self._notify_links([])
                finally:
                    self._notifier = None

    def _notify_link_list(self, links):
        if False:
            while True:
                i = 10
        if not links:
            return []
        only_while_ready = not self._notify_all
        final_link = links[-1]
        done = set()
        hub = self.hub if self.hub is not None else get_hub_if_exists()
        unswitched = []
        while links:
            if only_while_ready and (not self.ready()):
                break
            link = links.pop(0)
            id_link = id(link)
            if id_link not in done:
                done.add(id_link)
                try:
                    self._drop_lock_for_switch_out()
                    try:
                        link(self)
                    except greenlet_error:
                        unswitched.append(link)
                    finally:
                        self._acquire_lock_for_switch_in()
                except:
                    if hub is not None:
                        hub.handle_error((link, self), *sys.exc_info())
                    else:
                        import traceback
                        traceback.print_exc()
            if link is final_link:
                break
        return unswitched

    def _notify_links(self, arrived_while_waiting):
        if False:
            i = 10
            return i + 15
        notifier = self._notifier
        if notifier is None:
            self._check_and_notify()
            return
        try:
            unswitched = self._notify_link_list(self._links)
            if arrived_while_waiting:
                un2 = self._notify_link_list(arrived_while_waiting)
                unswitched.extend(un2)
                self._links.extend(arrived_while_waiting)
        finally:
            assert self._notifier is notifier, (self._notifier, notifier)
            self._notifier = None
        self._check_and_notify()
        if unswitched:
            self._handle_unswitched_notifications(unswitched)

    def _handle_unswitched_notifications(self, unswitched):
        if False:
            for i in range(10):
                print('nop')
        root_greenlets = None
        printed_tb = False
        only_while_ready = not self._notify_all
        while unswitched:
            if only_while_ready and (not self.ready()):
                self.__print_unswitched_warning(unswitched, printed_tb)
                break
            link = unswitched.pop(0)
            hub = None
            if getattr(link, '__name__', None) == 'switch' and isinstance(getattr(link, '__self__', None), greenlet):
                glet = link.__self__
                parent = glet.parent
                while parent is not None:
                    if hasattr(parent, 'loop'):
                        hub = glet.parent
                        break
                    parent = glet.parent
                if hub is None:
                    if root_greenlets is None:
                        root_greenlets = get_roots_and_hubs()
                    hub = root_greenlets.get(glet)
                if hub is not None and hub.loop is not None:
                    hub.loop.run_callback_threadsafe(link, self)
            if hub is None or hub.loop is None:
                self.__print_unswitched_warning(link, printed_tb)
                printed_tb = True

    def __print_unswitched_warning(self, link, printed_tb):
        if False:
            print('Hello World!')
        print('gevent: error: Unable to switch to greenlet', link, 'from', self, '; crossing thread boundaries is not allowed.', file=sys.stderr)
        if not printed_tb:
            printed_tb = True
            print('gevent: error: This is a result of using gevent objects from multiple threads,', 'and is a bug in the calling code.', file=sys.stderr)
            import traceback
            traceback.print_stack()

    def _quiet_unlink_all(self, obj):
        if False:
            print('Hello World!')
        if obj is None:
            return
        self.unlink(obj)
        if self._notifier is not None and self._notifier.args:
            try:
                self._notifier.args[0].remove(obj)
            except ValueError:
                pass

    def __wait_to_be_notified(self, rawlink):
        if False:
            print('Hello World!')
        resume_this_greenlet = getcurrent().switch
        if rawlink:
            self.rawlink(resume_this_greenlet)
        else:
            self._notifier.args[0].append(resume_this_greenlet)
        try:
            self._switch_to_hub(self.hub)
            resume_this_greenlet = None
        finally:
            self._quiet_unlink_all(resume_this_greenlet)

    def _switch_to_hub(self, the_hub):
        if False:
            while True:
                i = 10
        self._drop_lock_for_switch_out()
        try:
            result = the_hub.switch()
        finally:
            self._acquire_lock_for_switch_in()
        if result is not self:
            raise InvalidSwitchError('Invalid switch into %s.wait(): %r' % (self.__class__.__name__, result))

    def _acquire_lock_for_switch_in(self):
        if False:
            return 10
        return

    def _drop_lock_for_switch_out(self):
        if False:
            while True:
                i = 10
        return

    def _wait_core(self, timeout, catch=Timeout):
        if False:
            return 10
        '\n        The core of the wait implementation, handling switching and\n        linking.\n\n        This method is NOT safe to call from multiple threads.\n\n        ``self.hub`` must be initialized before entering this method.\n        The hub that is set is considered the owner and cannot be changed\n        while this method is running. It must only be called from the thread\n        where ``self.hub`` is the current hub.\n\n        If *catch* is set to ``()``, a timeout that elapses will be\n        allowed to be raised.\n\n        :return: A true value if the wait succeeded without timing out.\n          That is, a true return value means we were notified and control\n          resumed in this greenlet.\n        '
        with Timeout._start_new_or_dummy(timeout) as timer:
            try:
                self.__wait_to_be_notified(True)
                return True
            except catch as ex:
                if ex is not timer:
                    raise
                return False

    def _wait_return_value(self, waited, wait_success):
        if False:
            while True:
                i = 10
        return None

    def _wait(self, timeout=None):
        if False:
            print('Hello World!')
        self._capture_hub(True)
        if self.ready():
            result = self._wait_return_value(False, False)
            if self._notifier:
                self.__wait_to_be_notified(False)
            return result
        gotit = self._wait_core(timeout)
        return self._wait_return_value(True, gotit)

    def _at_fork_reinit(self):
        if False:
            return 10
        '\n        This method was added in Python 3.9 and is called by logging.py\n        ``_after_at_fork_child_reinit_locks`` on Lock objects.\n\n        It is also called from threading.py, ``_after_fork`` in\n        ``_reset_internal_locks``, and that can hit ``Event`` objects.\n\n        Subclasses should reset themselves to an initial state. This\n        includes unlocking/releasing, if possible. This method detaches from the\n        previous hub and drops any existing notifier.\n        '
        self.hub = None
        self._notifier = None

def _init():
    if False:
        i = 10
        return i + 15
    greenlet_init()
_init()
from gevent._util import import_c_accel
import_c_accel(globals(), 'gevent.__abstract_linkable')