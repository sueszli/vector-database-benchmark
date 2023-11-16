from __future__ import annotations
from typing import TYPE_CHECKING
from libqtile import config, group, hook
from libqtile.backend.base import FloatStates
from libqtile.command.base import expose_command
from libqtile.config import Match
if TYPE_CHECKING:
    from libqtile.backend.base import Window

class WindowVisibilityToggler:
    """
    WindowVisibilityToggler is a wrapper for a window, used in ScratchPad group
    to toggle visibility of a window by toggling the group it belongs to.
    The window is either sent to the named ScratchPad, which is by default
    invisble, or the current group on the current screen.
    With this functionality the window can be shown and hidden by a single
    keystroke (bound to command of ScratchPad group).
    By default, the window is also hidden if it loses focus.
    """

    def __init__(self, scratchpad_name, window: Window, on_focus_lost_hide, warp_pointer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initiliaze the  WindowVisibilityToggler.\n\n        Parameters:\n        ===========\n        scratchpad_name: string\n            The name (not label) of the ScratchPad group used to hide the window\n        window: window\n            The window to toggle\n        on_focus_lost_hide: bool\n            if True the associated window is hidden if it loses focus\n        warp_pointer: bool\n            if True the mouse pointer is warped to center of associated window\n            if shown. Only used if on_focus_lost_hide is True\n        '
        self.scratchpad_name = scratchpad_name
        self.window = window
        self.on_focus_lost_hide = on_focus_lost_hide
        self.warp_pointer = warp_pointer
        self.shown = False
        self.show()

    def info(self):
        if False:
            print('Hello World!')
        return dict(window=self.window.info(), scratchpad_name=self.scratchpad_name, visible=self.visible, on_focus_lost_hide=self.on_focus_lost_hide, warp_pointer=self.warp_pointer)

    @property
    def visible(self):
        if False:
            return 10
        '\n        Determine if associated window is currently visible.\n        That is the window is on a group different from the scratchpad\n        and that group is the current visible group.\n        '
        if self.window.group is None:
            return False
        return self.window.group.name != self.scratchpad_name and self.window.group is self.window.qtile.current_group

    def toggle(self):
        if False:
            print('Hello World!')
        '\n        Toggle the visibility of associated window. Either show() or hide().\n        '
        if not self.visible or not self.shown:
            self.show()
        else:
            self.hide()

    def show(self):
        if False:
            return 10
        "\n        Show the associated window on top of current screen.\n        The window is moved to the current group as floating window.\n\n        If 'warp_pointer' is True the mouse pointer is warped to center of the\n        window if 'on_focus_lost_hide' is True.\n        Otherwise, if pointer is moved manually to window by the user\n        the window might be hidden again before actually reaching it.\n        "
        if not self.visible or not self.shown:
            win = self.window
            win._float_state = FloatStates.TOP
            win.togroup()
            win.bring_to_front()
            self.shown = True
            if self.on_focus_lost_hide:
                if self.warp_pointer:
                    win.focus(warp=True)
                hook.subscribe.client_focus(self.on_focus_change)
                hook.subscribe.setgroup(self.on_focus_change)

    def hide(self):
        if False:
            while True:
                i = 10
        '\n        Hide the associated window. That is, send it to the scratchpad group.\n        '
        if self.visible or self.shown:
            if self.on_focus_lost_hide:
                hook.unsubscribe.client_focus(self.on_focus_change)
                hook.unsubscribe.setgroup(self.on_focus_change)
            self.window.togroup(self.scratchpad_name)
            self.shown = False

    def unsubscribe(self):
        if False:
            return 10
        'unsubscribe all hooks'
        if self.on_focus_lost_hide and (self.visible or self.shown):
            hook.unsubscribe.client_focus(self.on_focus_change)
            hook.unsubscribe.setgroup(self.on_focus_change)

    def on_focus_change(self, *args, **kwargs):
        if False:
            print('Hello World!')
        "\n        hook method which is called on window focus change and group change.\n        Depending on 'on_focus_lost_xxx' arguments, the associated window may\n        get hidden (by call to hide) or even killed.\n        "
        if self.shown:
            current_group = self.window.qtile.current_group
            if self.window.group is not current_group or self.window is not current_group.current_window:
                if self.on_focus_lost_hide:
                    self.hide()

class DropDownToggler(WindowVisibilityToggler):
    """
    Specialized WindowVisibilityToggler which places the associatd window
    each time it is shown at desired location.
    For example this can be used to create a quake-like terminal.
    """

    def __init__(self, window, scratchpad_name, ddconfig):
        if False:
            i = 10
            return i + 15
        self.name = ddconfig.name
        self.x = ddconfig.x
        self.y = ddconfig.y
        self.width = ddconfig.width
        self.height = ddconfig.height
        window.togroup(scratchpad_name)
        window.opacity = ddconfig.opacity
        WindowVisibilityToggler.__init__(self, scratchpad_name, window, ddconfig.on_focus_lost_hide, ddconfig.warp_pointer)

    def info(self):
        if False:
            for i in range(10):
                print('nop')
        info = WindowVisibilityToggler.info(self)
        info.update(dict(name=self.name, x=self.x, y=self.y, width=self.width, height=self.height))
        return info

    def show(self):
        if False:
            print('Hello World!')
        '\n        Like WindowVisibilityToggler.show, but before showing the window,\n        its floating x, y, width and height is set.\n        '
        if not self.visible or not self.shown:
            win = self.window
            screen = win.qtile.current_screen
            x = int(screen.dx + self.x * screen.dwidth)
            y = int(screen.dy + self.y * screen.dheight)
            win.float_x = x
            win.float_y = y
            width = int(screen.dwidth * self.width)
            height = int(screen.dheight * self.height)
            win.place(x, y, width, height, win.borderwidth, win.bordercolor, respect_hints=True)
            WindowVisibilityToggler.show(self)

class ScratchPad(group._Group):
    """
    Specialized group which is by default invisible and can be configured, to
    spawn windows and toggle its visibility (in the current group) by command.

    The ScratchPad group acts as a container for windows which are currently
    not visible but associated to a DropDownToggler and can toggle their
    group by command (of ScratchPad group).
    The ScratchPad, by default, has no label and thus is not shown in
    GroupBox widget.
    """

    def __init__(self, name='scratchpad', dropdowns: list[config.DropDown] | None=None, label='', single=False):
        if False:
            while True:
                i = 10
        group._Group.__init__(self, name, label=label)
        self._dropdownconfig = {dd.name: dd for dd in dropdowns} if dropdowns is not None else {}
        self.dropdowns: dict[str, DropDownToggler] = {}
        self._spawned: dict[str, Match] = {}
        self._to_hide: list[str] = []
        self._single = single

    def _check_unsubscribe(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.dropdowns:
            hook.unsubscribe.client_killed(self.on_client_killed)
            hook.unsubscribe.float_change(self.on_float_change)

    def _spawn(self, ddconfig):
        if False:
            while True:
                i = 10
        '\n        Spawn a process by defined command.\n        Method is only called if no window is associated. This is either on the\n        first call to show or if the window was killed.\n        The process id of spawned process is saved and compared to new windows.\n        In case of a match the window gets associated to this DropDown object.\n        '
        name = ddconfig.name
        if name not in self._spawned:
            if not self._spawned:
                hook.subscribe.client_new(self.on_client_new)
            pid = self.qtile.spawn(ddconfig.command)
            self._spawned[name] = ddconfig.match or Match(net_wm_pid=pid)

    def on_client_new(self, client, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        hook method which is called on new windows.\n        This method is subscribed if the given command is spawned\n        and unsubscribed immediately if the associated window is detected.\n        '
        name = None
        for (n, match) in self._spawned.items():
            if match.compare(client):
                name = n
                break
        if name is not None:
            self._spawned.pop(name)
            if not self._spawned:
                hook.unsubscribe.client_new(self.on_client_new)
            self.dropdowns[name] = DropDownToggler(client, self.name, self._dropdownconfig[name])
            if self._single:
                for (n, d) in self.dropdowns.items():
                    if n != name:
                        d.hide()
            if name in self._to_hide:
                self.dropdowns[name].hide()
                self._to_hide.remove(name)
            if len(self.dropdowns) == 1:
                hook.subscribe.client_killed(self.on_client_killed)
                hook.subscribe.float_change(self.on_float_change)

    def on_client_killed(self, client, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        hook method which is called if a client is killed.\n        If the associated window is killed, reset internal state.\n        '
        name = None
        for (name, dd) in self.dropdowns.items():
            if dd.window is client:
                del self.dropdowns[name]
                break
        self._check_unsubscribe()

    def on_float_change(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        hook method which is called if window float state is changed.\n        If the current associated window is not floated (any more) the window\n        and process is detached from DRopDown, thus the next call to Show\n        will spawn a new process.\n        '
        name = None
        for (name, dd) in self.dropdowns.items():
            if not dd.window.floating:
                if dd.window.group is not self:
                    dd.unsubscribe()
                    del self.dropdowns[name]
                    break
        self._check_unsubscribe()

    @expose_command()
    def dropdown_toggle(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Toggle visibility of named DropDown.\n        '
        if self._single:
            for (n, d) in self.dropdowns.items():
                if n != name:
                    d.hide()
        if name in self.dropdowns:
            self.dropdowns[name].toggle()
        elif name in self._dropdownconfig:
            self._spawn(self._dropdownconfig[name])

    @expose_command()
    def hide_all(self):
        if False:
            print('Hello World!')
        '\n        Hide all scratchpads.\n        '
        for d in self.dropdowns.values():
            d.hide()

    @expose_command()
    def dropdown_reconfigure(self, name, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        reconfigure the named DropDown configuration.\n        Note that changed attributes only have an effect on spawning the window.\n        '
        if name not in self._dropdownconfig:
            return
        dd = self._dropdownconfig[name]
        for (attr, value) in kwargs.items():
            if hasattr(dd, attr):
                setattr(dd, attr, value)

    @expose_command()
    def dropdown_info(self, name=None):
        if False:
            print('Hello World!')
        '\n        Get information on configured or currently active DropDowns.\n        If name is None, a list of all dropdown names is returned.\n        '
        if name is None:
            return {'dropdowns': [ddname for ddname in self._dropdownconfig]}
        elif name in self.dropdowns:
            return self.dropdowns[name].info()
        elif name in self._dropdownconfig:
            return self._dropdownconfig[name].info()
        else:
            raise ValueError('No DropDown named "%s".' % name)

    def get_state(self):
        if False:
            return 10
        '\n        Get the state of existing dropdown windows. Used for restoring state across\n        Qtile restarts (`restart` == True) or config reloads (`restart` == False).\n        '
        state = []
        for (name, dd) in self.dropdowns.items():
            client_wid = dd.window.wid
            state.append((name, client_wid, dd.visible))
        return state

    def restore_state(self, state, restart: bool) -> list[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Restore the state of existing dropdown windows. Used for restoring state across\n        Qtile restarts (`restart` == True) or config reloads (`restart` == False).\n        '
        orphans = []
        for (name, wid, visible) in state:
            if name in self._dropdownconfig:
                if restart:
                    self._spawned[name] = Match(wid=wid)
                    if not visible:
                        self._to_hide.append(name)
                else:
                    self.dropdowns[name] = DropDownToggler(self.qtile.windows_map[wid], self.name, self._dropdownconfig[name])
                    if not visible:
                        self.dropdowns[name].hide()
            else:
                orphans.append(wid)
        if self._spawned:
            assert restart
            hook.subscribe.client_new(self.on_client_new)
        if not restart and self.dropdowns:
            hook.subscribe.client_killed(self.on_client_killed)
            hook.subscribe.float_change(self.on_float_change)
        return orphans