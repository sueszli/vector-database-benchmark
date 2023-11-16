import os
from functools import partial
from typing import Callable, List, Optional
import cairocffi
from dbus_next import InterfaceNotFoundError, InvalidBusNameError, InvalidObjectPathError
from dbus_next.aio import MessageBus
from dbus_next.constants import PropertyAccess
from dbus_next.errors import DBusError
from dbus_next.service import ServiceInterface, dbus_property, method, signal
try:
    from xdg.IconTheme import getIconPath
    has_xdg = True
except ImportError:
    has_xdg = False
from libqtile import bar
from libqtile.images import Img
from libqtile.log_utils import logger
from libqtile.resources.status_notifier.statusnotifieritem import STATUS_NOTIFIER_ITEM_SPEC
from libqtile.utils import add_signal_receiver, create_task
from libqtile.widget import base
BUS_NAMES = ['org.kde.StatusNotifierWatcher', 'org.freedesktop.StatusNotifierWatcher']
ITEM_INTERFACES = ['org.kde.StatusNotifierItem', 'org.freedesktop.StatusNotifierItem']
STATUSNOTIFIER_PATH = '/StatusNotifierItem'
PROTOCOL_VERSION = 0

class StatusNotifierItem:
    """
    Class object which represents an StatusNotiferItem object.

    The item is responsible for interacting with the
    application.
    """
    icon_map = {'Icon': ('_icon', 'get_icon_pixmap'), 'Attention': ('_attention_icon', 'get_attention_icon_pixmap'), 'Overlay': ('_overlay_icon', 'get_overlay_icon_pixmap')}

    def __init__(self, bus, service, path=None, icon_theme=None):
        if False:
            while True:
                i = 10
        self.bus = bus
        self.service = service
        self.surfaces = {}
        self._pixmaps = {}
        self._icon = None
        self._overlay_icon = None
        self._attention_icon = None
        self.on_icon_changed = None
        self.icon_theme = icon_theme
        self.icon = None
        self.path = path if path else STATUSNOTIFIER_PATH

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, StatusNotifierItem):
            return other.service == self.service
        elif isinstance(other, str):
            return other == self.service
        else:
            return False

    async def start(self):
        found_path = False
        while not found_path:
            try:
                introspection = await self.bus.introspect(self.service, self.path)
                found_path = True
            except InvalidBusNameError:
                return False
            except InvalidObjectPathError:
                logger.info('Cannot find %s path on %s.', self.path, self.service)
                if self.path == STATUSNOTIFIER_PATH:
                    return False
                self.path = STATUSNOTIFIER_PATH
        try:
            obj = self.bus.get_proxy_object(self.service, self.path, introspection)
        except InvalidBusNameError:
            return False
        interface_found = False
        for interface in ITEM_INTERFACES:
            try:
                self.item = obj.get_interface(interface)
                interface_found = True
                break
            except InterfaceNotFoundError:
                continue
        if not interface_found:
            logger.info('Unable to find StatusNotifierItem interface on %s. Falling back to default spec.', self.service)
            try:
                obj = self.bus.get_proxy_object(self.service, STATUSNOTIFIER_PATH, STATUS_NOTIFIER_ITEM_SPEC)
                self.item = obj.get_interface('org.kde.StatusNotifierItem')
            except InterfaceNotFoundError:
                logger.warning('Failed to find StatusNotifierItem interface on %s and fallback to default spec also failed.', self.service)
                return False
        await self._get_local_icon()
        if self.icon:
            self.item.on_new_icon(self._update_local_icon)
        else:
            for icon in ['Icon', 'Attention', 'Overlay']:
                await self._get_icon(icon)
            if self.has_icons:
                self.item.on_new_icon(self._new_icon)
                self.item.on_new_attention_icon(self._new_attention_icon)
                self.item.on_new_overlay_icon(self._new_overlay_icon)
        if not self.has_icons:
            logger.warning('Cannot find icon in current theme and no icon provided by StatusNotifierItem.')
        return True

    async def _get_local_icon(self):
        try:
            icon_name = await self.item.get_icon_name()
        except DBusError:
            return
        try:
            icon_path = await self.item.get_icon_theme_path()
            self.icon = self._get_custom_icon(icon_name, icon_path)
        except (AttributeError, DBusError):
            pass
        if not self.icon:
            self.icon = self._get_xdg_icon(icon_name)

    def _create_task_and_draw(self, coro):
        if False:
            return 10
        task = create_task(coro)
        task.add_done_callback(self._redraw)

    def _update_local_icon(self):
        if False:
            while True:
                i = 10
        self.icon = None
        self._create_task_and_draw(self._get_local_icon())

    def _new_icon(self):
        if False:
            i = 10
            return i + 15
        self._create_task_and_draw(self._get_icon('Icon'))

    def _new_attention_icon(self):
        if False:
            for i in range(10):
                print('nop')
        self._create_task_and_draw(self._get_icon('Attention'))

    def _new_overlay_icon(self):
        if False:
            print('Hello World!')
        self._create_task_and_draw(self._get_icon('Overlay'))

    def _get_custom_icon(self, icon_name, icon_path):
        if False:
            i = 10
            return i + 15
        for ext in ['.png', '.svg']:
            path = os.path.join(icon_path, icon_name + ext)
            if os.path.isfile(path):
                return Img.from_path(path)
        return None

    def _get_xdg_icon(self, icon_name):
        if False:
            while True:
                i = 10
        if not has_xdg:
            return
        path = getIconPath(icon_name, theme=self.icon_theme, extensions=['png', 'svg'])
        if not path:
            return None
        return Img.from_path(path)

    async def _get_icon(self, icon_name):
        """
        Requests the pixmap for the given `icon_name` and
        adds to an internal dictionary for later retrieval.
        """
        (attr, method) = self.icon_map[icon_name]
        pixmap = getattr(self.item, method, None)
        if pixmap is None:
            return
        icon_pixmap = await pixmap()
        self._pixmaps[icon_name] = {size: self._reorder_bytes(icon_bytes) for (size, _, icon_bytes) in icon_pixmap}

    def _reorder_bytes(self, icon_bytes):
        if False:
            print('Hello World!')
        '\n        Method loops over the array and reverses every\n        4 bytes (representing one RGBA pixel).\n        '
        arr = bytearray(icon_bytes)
        for i in range(0, len(arr), 4):
            arr[i:i + 4] = arr[i:i + 4][::-1]
        return arr

    def _redraw(self, result):
        if False:
            print('Hello World!')
        'Method to invalidate icon cache and redraw icons.'
        self._invalidate_icons()
        if self.on_icon_changed is not None:
            self.on_icon_changed(self)

    def _invalidate_icons(self):
        if False:
            i = 10
            return i + 15
        self.surfaces = {}

    def _get_sizes(self):
        if False:
            i = 10
            return i + 15
        'Returns list of available icon sizes.'
        if not self._pixmaps.get('Icon', False):
            return []
        return sorted([size for size in self._pixmaps['Icon']])

    def _get_surfaces(self, size):
        if False:
            i = 10
            return i + 15
        '\n        Creates a Cairo ImageSurface for each available icon\n        for the given size.\n        '
        raw_surfaces = {}
        for icon in self._pixmaps:
            if size in self._pixmaps[icon]:
                srf = cairocffi.ImageSurface.create_for_data(self._pixmaps[icon][size], cairocffi.FORMAT_ARGB32, size, size)
                raw_surfaces[icon] = srf
        return raw_surfaces

    def get_icon(self, size):
        if False:
            return 10
        '\n        Returns a cairo ImageSurface for the selected `size`.\n\n        Will pick the appropriate icon and add any overlay as required.\n        '
        if size in self.surfaces:
            return self.surfaces[size]
        icon = cairocffi.ImageSurface(cairocffi.FORMAT_ARGB32, size, size)
        if self.icon:
            base_icon = self.icon.surface
            icon_size = base_icon.get_width()
            overlay = None
        else:
            all_sizes = self._get_sizes()
            sizes = [s for s in all_sizes if s >= size]
            if not all_sizes:
                return icon
            icon_size = sizes[0] if sizes else all_sizes[-1]
            srfs = self._get_surfaces(icon_size)
            if not srfs:
                return icon
            base_icon = srfs.get('Attention', srfs['Icon'])
            overlay = srfs.get('Overlay', None)
        with cairocffi.Context(icon) as ctx:
            scale = size / icon_size
            ctx.scale(scale, scale)
            ctx.set_source_surface(base_icon)
            ctx.paint()
            if overlay:
                ctx.set_source_surface(overlay)
                ctx.paint()
        self.surfaces[size] = icon
        return icon

    def activate(self):
        if False:
            while True:
                i = 10
        if hasattr(self.item, 'call_activate'):
            create_task(self._activate())

    async def _activate(self):
        await self.item.call_activate(0, 0)

    @property
    def has_icons(self):
        if False:
            i = 10
            return i + 15
        return any((bool(icon) for icon in self._pixmaps.values())) or self.icon is not None

class StatusNotifierWatcher(ServiceInterface):
    """
    DBus service that creates a StatusNotifierWatcher interface
    on the bus and listens for applications wanting to register
    items.
    """

    def __init__(self, service: str):
        if False:
            while True:
                i = 10
        super().__init__(service)
        self._items: List[str] = []
        self._hosts: List[str] = []
        self.service = service
        self.on_item_added: Optional[Callable] = None
        self.on_host_added: Optional[Callable] = None
        self.on_item_removed: Optional[Callable] = None
        self.on_host_removed: Optional[Callable] = None

    async def start(self):
        self.bus = await MessageBus().connect()
        self.bus.add_message_handler(self._message_handler)
        self.bus.export('/StatusNotifierWatcher', self)
        await self.bus.request_name(self.service)
        await self._setup_listeners()

    def _message_handler(self, message):
        if False:
            return 10
        '\n        Low level method to check incoming messages.\n\n        Ayatana indicators seem to register themselves by passing their object\n        path rather than the service providing that object. We therefore need\n        to identify the sender of the message in order to register the service.\n\n        Returning False so senders receieve a reply (returning True prevents\n        reply being sent)\n        '
        if message.member != 'RegisterStatusNotifierItem':
            return False
        if not message.body[0].startswith('/'):
            return False
        if message.sender not in self._items:
            self._items.append(message.sender)
            if self.on_item_added is not None:
                self.on_item_added(message.sender, message.body[0])
            self.StatusNotifierItemRegistered(message.sender)
        return False

    async def _setup_listeners(self):
        """
        Register a MatchRule to receive signals when interfaces are added
        and removed from the bus.
        """
        await add_signal_receiver(self._name_owner_changed, session_bus=True, signal_name='NameOwnerChanged', dbus_interface='org.freedesktop.DBus')

    def _name_owner_changed(self, message):
        if False:
            return 10
        (name, _, new_owner) = message.body
        if new_owner == '' and name in self._items:
            self._items.remove(name)
            self.StatusNotifierItemUnregistered(name)
        if new_owner == '' and name in self._hosts:
            self._hosts.remove(name)
            self.StatusNotifierHostUnregistered(name)

    @method()
    def RegisterStatusNotifierItem(self, service: 's'):
        if False:
            i = 10
            return i + 15
        if service not in self._items:
            self._items.append(service)
            if self.on_item_added is not None:
                self.on_item_added(service)
            self.StatusNotifierItemRegistered(service)

    @method()
    def RegisterStatusNotifierHost(self, service: 's'):
        if False:
            while True:
                i = 10
        if service not in self._hosts:
            self._hosts.append(service)
            self.StatusNotifierHostRegistered(service)

    @dbus_property(access=PropertyAccess.READ)
    def RegisteredStatusNotifierItems(self) -> 'as':
        if False:
            print('Hello World!')
        return self._items

    @dbus_property(access=PropertyAccess.READ)
    def IsStatusNotifierHostRegistered(self) -> 'b':
        if False:
            for i in range(10):
                print('nop')
        return len(self._hosts) > 0

    @dbus_property(access=PropertyAccess.READ)
    def ProtocolVersion(self) -> 'i':
        if False:
            while True:
                i = 10
        return PROTOCOL_VERSION

    @signal()
    def StatusNotifierItemRegistered(self, service) -> 's':
        if False:
            for i in range(10):
                print('nop')
        return service

    @signal()
    def StatusNotifierItemUnregistered(self, service) -> 's':
        if False:
            while True:
                i = 10
        if self.on_item_removed is not None:
            self.on_item_removed(service)
        return service

    @signal()
    def StatusNotifierHostRegistered(self, service) -> 's':
        if False:
            print('Hello World!')
        if self.on_host_added is not None:
            self.on_host_added(service)
        return service

    @signal()
    def StatusNotifierHostUnregistered(self, service) -> 's':
        if False:
            while True:
                i = 10
        if self.on_host_removed is not None:
            self.on_host_removed(service)
        return service

class StatusNotifierHost:
    """
    Host object to act as a bridge between the widget and the DBus objects.

    The Host collates items returned from multiple watcher interfaces and
    collates them into a single list for the widget to access.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.watchers: List[StatusNotifierWatcher] = []
        self.items: List[StatusNotifierItem] = []
        self.name = 'qtile'
        self.icon_theme: str = None
        self.started = False
        self._on_item_added: List[Callable] = []
        self._on_item_removed: List[Callable] = []
        self._on_icon_changed: List[Callable] = []

    async def start(self, on_item_added: Optional[Callable]=None, on_item_removed: Optional[Callable]=None, on_icon_changed: Optional[Callable]=None):
        """
        Starts the host if not already started.

        Widgets should register their callbacks via this method.
        """
        if on_item_added:
            self._on_item_added.append(on_item_added)
        if on_item_removed:
            self._on_item_removed.append(on_item_removed)
        if on_icon_changed:
            self._on_icon_changed.append(on_icon_changed)
        if self.started:
            if on_item_added:
                for item in self.items:
                    on_item_added(item)
            return
        self.bus = await MessageBus().connect()
        await self.bus.request_name('org.freedesktop.StatusNotifierHost-qtile')
        for iface in BUS_NAMES:
            w = StatusNotifierWatcher(iface)
            w.on_item_added = self.add_item
            w.on_item_removed = self.remove_item
            await w.start()
            w.RegisterStatusNotifierHost(self.name)
            self.watchers.append(w)
            self.started = True

    def item_added(self, item, service, future):
        if False:
            print('Hello World!')
        success = future.result()
        if success:
            self.items.append(item)
            for callback in self._on_item_added:
                callback(item)
        else:
            for w in self.watchers:
                try:
                    w._items.remove(service)
                except ValueError:
                    pass

    def add_item(self, service, path=None):
        if False:
            print('Hello World!')
        '\n        Creates a StatusNotifierItem for the given service and tries to\n        start it.\n        '
        item = StatusNotifierItem(self.bus, service, path=path, icon_theme=self.icon_theme)
        item.on_icon_changed = self.item_icon_changed
        if item not in self.items:
            task = create_task(item.start())
            task.add_done_callback(partial(self.item_added, item, service))

    def remove_item(self, interface):
        if False:
            print('Hello World!')
        if interface in self.items:
            self.items.remove(interface)
            for callback in self._on_item_removed:
                callback(interface)

    def item_icon_changed(self, item):
        if False:
            for i in range(10):
                print('nop')
        for callback in self._on_icon_changed:
            callback(item)
host = StatusNotifierHost()

class StatusNotifier(base._Widget):
    """
    A 'system tray' widget using the freedesktop StatusNotifierItem
    specification.

    As per the specification, app icons are first retrieved from the
    user's current theme. If this is not available then the app may
    provide its own icon. In order to use this functionality, users
    are recommended to install the `pyxdg <https://pypi.org/project/pyxdg/>`__
    module to support retrieving icons from the selected theme.

    Left-clicking an icon will trigger an activate event.

    .. note::

        Context menus are not currently supported by the official widget.
        However, a modded version of the widget which provides basic menu
        support is available from elParaguayo's `qtile-extras
        <https://github.com/elParaguayo/qtile-extras>`_ repo.
    """
    orientations = base.ORIENTATION_BOTH
    defaults = [('icon_size', 16, 'Icon width'), ('icon_theme', None, 'Name of theme to use for app icons'), ('padding', 3, 'Padding between icons')]

    def __init__(self, **config):
        if False:
            print('Hello World!')
        base._Widget.__init__(self, bar.CALCULATED, **config)
        self.add_defaults(StatusNotifier.defaults)
        self.add_callbacks({'Button1': self.activate})
        self.selected_item: Optional[StatusNotifierItem] = None

    @property
    def available_icons(self):
        if False:
            for i in range(10):
                print('nop')
        return [item for item in host.items if item.has_icons]

    def calculate_length(self):
        if False:
            return 10
        if not host.items:
            return 0
        return len(self.available_icons) * (self.icon_size + self.padding) + self.padding

    def _configure(self, qtile, bar):
        if False:
            while True:
                i = 10
        if has_xdg and self.icon_theme:
            host.icon_theme = self.icon_theme
        base._Widget._configure(self, qtile, bar)

    async def _config_async(self):

        def draw(x=None):
            if False:
                return 10
            self.bar.draw()
        await host.start(on_item_added=draw, on_item_removed=draw, on_icon_changed=draw)

    def find_icon_at_pos(self, x, y):
        if False:
            i = 10
            return i + 15
        'returns StatusNotifierItem object for icon in given position'
        offset = self.padding
        val = x if self.bar.horizontal else y
        if val < offset:
            return None
        for icon in self.available_icons:
            offset += self.icon_size
            if val < offset:
                return icon
            offset += self.padding
        return None

    def button_press(self, x, y, button):
        if False:
            return 10
        icon = self.find_icon_at_pos(x, y)
        self.selected_item = icon if icon else None
        name = 'Button{0}'.format(button)
        if name in self.mouse_callbacks:
            self.mouse_callbacks[name]()

    def draw(self):
        if False:
            while True:
                i = 10
        self.drawer.clear(self.background or self.bar.background)
        if self.bar.horizontal:
            xoffset = self.padding
            yoffset = (self.bar.height - self.icon_size) // 2
            for item in self.available_icons:
                icon = item.get_icon(self.icon_size)
                self.drawer.ctx.set_source_surface(icon, xoffset, yoffset)
                self.drawer.ctx.paint()
                xoffset += self.icon_size + self.padding
            self.drawer.draw(offsetx=self.offset, offsety=self.offsety, width=self.length)
        else:
            xoffset = (self.bar.width - self.icon_size) // 2
            yoffset = self.padding
            for item in self.available_icons:
                icon = item.get_icon(self.icon_size)
                self.drawer.ctx.set_source_surface(icon, xoffset, yoffset)
                self.drawer.ctx.paint()
                yoffset += self.icon_size + self.padding
            self.drawer.draw(offsety=self.offset, offsetx=self.offsetx, height=self.length)

    def activate(self):
        if False:
            return 10
        'Primary action when clicking on an icon'
        if not self.selected_item:
            return
        self.selected_item.activate()