from matplotlib import _api, backend_tools, cbook, widgets

class ToolEvent:
    """Event for tool manipulation (add/remove)."""

    def __init__(self, name, sender, tool, data=None):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.sender = sender
        self.tool = tool
        self.data = data

class ToolTriggerEvent(ToolEvent):
    """Event to inform that a tool has been triggered."""

    def __init__(self, name, sender, tool, canvasevent=None, data=None):
        if False:
            return 10
        super().__init__(name, sender, tool, data)
        self.canvasevent = canvasevent

class ToolManagerMessageEvent:
    """
    Event carrying messages from toolmanager.

    Messages usually get displayed to the user by the toolbar.
    """

    def __init__(self, name, sender, message):
        if False:
            return 10
        self.name = name
        self.sender = sender
        self.message = message

class ToolManager:
    """
    Manager for actions triggered by user interactions (key press, toolbar
    clicks, ...) on a Figure.

    Attributes
    ----------
    figure : `.Figure`
    keypresslock : `~matplotlib.widgets.LockDraw`
        `.LockDraw` object to know if the `canvas` key_press_event is locked.
    messagelock : `~matplotlib.widgets.LockDraw`
        `.LockDraw` object to know if the message is available to write.
    """

    def __init__(self, figure=None):
        if False:
            while True:
                i = 10
        self._key_press_handler_id = None
        self._tools = {}
        self._keys = {}
        self._toggled = {}
        self._callbacks = cbook.CallbackRegistry()
        self.keypresslock = widgets.LockDraw()
        self.messagelock = widgets.LockDraw()
        self._figure = None
        self.set_figure(figure)

    @property
    def canvas(self):
        if False:
            i = 10
            return i + 15
        'Canvas managed by FigureManager.'
        if not self._figure:
            return None
        return self._figure.canvas

    @property
    def figure(self):
        if False:
            print('Hello World!')
        'Figure that holds the canvas.'
        return self._figure

    @figure.setter
    def figure(self, figure):
        if False:
            return 10
        self.set_figure(figure)

    def set_figure(self, figure, update_tools=True):
        if False:
            return 10
        '\n        Bind the given figure to the tools.\n\n        Parameters\n        ----------\n        figure : `.Figure`\n        update_tools : bool, default: True\n            Force tools to update figure.\n        '
        if self._key_press_handler_id:
            self.canvas.mpl_disconnect(self._key_press_handler_id)
        self._figure = figure
        if figure:
            self._key_press_handler_id = self.canvas.mpl_connect('key_press_event', self._key_press)
        if update_tools:
            for tool in self._tools.values():
                tool.figure = figure

    def toolmanager_connect(self, s, func):
        if False:
            i = 10
            return i + 15
        "\n        Connect event with string *s* to *func*.\n\n        Parameters\n        ----------\n        s : str\n            The name of the event. The following events are recognized:\n\n            - 'tool_message_event'\n            - 'tool_removed_event'\n            - 'tool_added_event'\n\n            For every tool added a new event is created\n\n            - 'tool_trigger_TOOLNAME', where TOOLNAME is the id of the tool.\n\n        func : callable\n            Callback function for the toolmanager event with signature::\n\n                def func(event: ToolEvent) -> Any\n\n        Returns\n        -------\n        cid\n            The callback id for the connection. This can be used in\n            `.toolmanager_disconnect`.\n        "
        return self._callbacks.connect(s, func)

    def toolmanager_disconnect(self, cid):
        if False:
            print('Hello World!')
        "\n        Disconnect callback id *cid*.\n\n        Example usage::\n\n            cid = toolmanager.toolmanager_connect('tool_trigger_zoom', onpress)\n            #...later\n            toolmanager.toolmanager_disconnect(cid)\n        "
        return self._callbacks.disconnect(cid)

    def message_event(self, message, sender=None):
        if False:
            return 10
        'Emit a `ToolManagerMessageEvent`.'
        if sender is None:
            sender = self
        s = 'tool_message_event'
        event = ToolManagerMessageEvent(s, sender, message)
        self._callbacks.process(s, event)

    @property
    def active_toggle(self):
        if False:
            while True:
                i = 10
        'Currently toggled tools.'
        return self._toggled

    def get_tool_keymap(self, name):
        if False:
            print('Hello World!')
        '\n        Return the keymap associated with the specified tool.\n\n        Parameters\n        ----------\n        name : str\n            Name of the Tool.\n\n        Returns\n        -------\n        list of str\n            List of keys associated with the tool.\n        '
        keys = [k for (k, i) in self._keys.items() if i == name]
        return keys

    def _remove_keys(self, name):
        if False:
            while True:
                i = 10
        for k in self.get_tool_keymap(name):
            del self._keys[k]

    def update_keymap(self, name, key):
        if False:
            return 10
        '\n        Set the keymap to associate with the specified tool.\n\n        Parameters\n        ----------\n        name : str\n            Name of the Tool.\n        key : str or list of str\n            Keys to associate with the tool.\n        '
        if name not in self._tools:
            raise KeyError(f'{name!r} not in Tools')
        self._remove_keys(name)
        if isinstance(key, str):
            key = [key]
        for k in key:
            if k in self._keys:
                _api.warn_external(f'Key {k} changed from {self._keys[k]} to {name}')
            self._keys[k] = name

    def remove_tool(self, name):
        if False:
            i = 10
            return i + 15
        '\n        Remove tool named *name*.\n\n        Parameters\n        ----------\n        name : str\n            Name of the tool.\n        '
        tool = self.get_tool(name)
        if getattr(tool, 'toggled', False):
            self.trigger_tool(tool, 'toolmanager')
        self._remove_keys(name)
        event = ToolEvent('tool_removed_event', self, tool)
        self._callbacks.process(event.name, event)
        del self._tools[name]

    def add_tool(self, name, tool, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Add *tool* to `ToolManager`.\n\n        If successful, adds a new event ``tool_trigger_{name}`` where\n        ``{name}`` is the *name* of the tool; the event is fired every time the\n        tool is triggered.\n\n        Parameters\n        ----------\n        name : str\n            Name of the tool, treated as the ID, has to be unique.\n        tool : type\n            Class of the tool to be added.  A subclass will be used\n            instead if one was registered for the current canvas class.\n        *args, **kwargs\n            Passed to the *tool*'s constructor.\n\n        See Also\n        --------\n        matplotlib.backend_tools.ToolBase : The base class for tools.\n        "
        tool_cls = backend_tools._find_tool_class(type(self.canvas), tool)
        if not tool_cls:
            raise ValueError('Impossible to find class for %s' % str(tool))
        if name in self._tools:
            _api.warn_external('A "Tool class" with the same name already exists, not added')
            return self._tools[name]
        tool_obj = tool_cls(self, name, *args, **kwargs)
        self._tools[name] = tool_obj
        if tool_obj.default_keymap is not None:
            self.update_keymap(name, tool_obj.default_keymap)
        if isinstance(tool_obj, backend_tools.ToolToggleBase):
            if tool_obj.radio_group is None:
                self._toggled.setdefault(None, set())
            else:
                self._toggled.setdefault(tool_obj.radio_group, None)
            if tool_obj.toggled:
                self._handle_toggle(tool_obj, None, None)
        tool_obj.set_figure(self.figure)
        event = ToolEvent('tool_added_event', self, tool_obj)
        self._callbacks.process(event.name, event)
        return tool_obj

    def _handle_toggle(self, tool, canvasevent, data):
        if False:
            while True:
                i = 10
        '\n        Toggle tools, need to untoggle prior to using other Toggle tool.\n        Called from trigger_tool.\n\n        Parameters\n        ----------\n        tool : `.ToolBase`\n        canvasevent : Event\n            Original Canvas event or None.\n        data : object\n            Extra data to pass to the tool when triggering.\n        '
        radio_group = tool.radio_group
        if radio_group is None:
            if tool.name in self._toggled[None]:
                self._toggled[None].remove(tool.name)
            else:
                self._toggled[None].add(tool.name)
            return
        if self._toggled[radio_group] == tool.name:
            toggled = None
        elif self._toggled[radio_group] is None:
            toggled = tool.name
        else:
            self.trigger_tool(self._toggled[radio_group], self, canvasevent, data)
            toggled = tool.name
        self._toggled[radio_group] = toggled

    def trigger_tool(self, name, sender=None, canvasevent=None, data=None):
        if False:
            while True:
                i = 10
        '\n        Trigger a tool and emit the ``tool_trigger_{name}`` event.\n\n        Parameters\n        ----------\n        name : str\n            Name of the tool.\n        sender : object\n            Object that wishes to trigger the tool.\n        canvasevent : Event\n            Original Canvas event or None.\n        data : object\n            Extra data to pass to the tool when triggering.\n        '
        tool = self.get_tool(name)
        if tool is None:
            return
        if sender is None:
            sender = self
        if isinstance(tool, backend_tools.ToolToggleBase):
            self._handle_toggle(tool, canvasevent, data)
        tool.trigger(sender, canvasevent, data)
        s = 'tool_trigger_%s' % name
        event = ToolTriggerEvent(s, sender, tool, canvasevent, data)
        self._callbacks.process(s, event)

    def _key_press(self, event):
        if False:
            print('Hello World!')
        if event.key is None or self.keypresslock.locked():
            return
        name = self._keys.get(event.key, None)
        if name is None:
            return
        self.trigger_tool(name, canvasevent=event)

    @property
    def tools(self):
        if False:
            i = 10
            return i + 15
        'A dict mapping tool name -> controlled tool.'
        return self._tools

    def get_tool(self, name, warn=True):
        if False:
            i = 10
            return i + 15
        '\n        Return the tool object with the given name.\n\n        For convenience, this passes tool objects through.\n\n        Parameters\n        ----------\n        name : str or `.ToolBase`\n            Name of the tool, or the tool itself.\n        warn : bool, default: True\n            Whether a warning should be emitted it no tool with the given name\n            exists.\n\n        Returns\n        -------\n        `.ToolBase` or None\n            The tool or None if no tool with the given name exists.\n        '
        if isinstance(name, backend_tools.ToolBase) and name.name in self._tools:
            return name
        if name not in self._tools:
            if warn:
                _api.warn_external(f'ToolManager does not control tool {name!r}')
            return None
        return self._tools[name]