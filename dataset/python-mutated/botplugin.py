import logging
import re
import shlex
from io import IOBase
from threading import Timer, current_thread
from types import ModuleType
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple
from errbot.backends.base import ONLINE, Card, Identifier, Message, Presence, Reaction, Room, Stream
from .storage import StoreMixin, StoreNotOpenError
log = logging.getLogger(__name__)

class ValidationException(Exception):
    pass

def recurse_check_structure(sample: Any, to_check: Any) -> None:
    if False:
        print('Hello World!')
    sample_type = type(sample)
    to_check_type = type(to_check)
    if sample is not None and sample_type != to_check_type:
        raise ValidationException(f'{sample} [{sample_type}] is not the same type as {to_check} [{to_check_type}].')
    if sample_type in (list, tuple):
        for element in to_check:
            recurse_check_structure(sample[0], element)
        return
    if sample_type == dict:
        for key in sample:
            if key not in to_check:
                raise ValidationException(f"{to_check} doesn't contain the key {key}.")
        for key in to_check:
            if key not in sample:
                raise ValidationException(f'{to_check} contains an unknown key {key}.')
        for key in sample:
            recurse_check_structure(sample[key], to_check[key])
        return

class CommandError(Exception):
    """
    Use this class to report an error condition from your commands, the command
    did not proceed for a known "business" reason.
    """

    def __init__(self, reason: str, template: str=None):
        if False:
            print('Hello World!')
        '\n        :param reason: the reason for the error in the command.\n        :param template: apply this specific template to report the error.\n        '
        self.reason = reason
        self.template = template

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.reason)

class Command:
    """
    This is a dynamic definition of an errbot command.
    """

    def __init__(self, function: Callable, cmd_type: Optional[Callable]=None, cmd_args=None, cmd_kwargs=None, name: Optional[str]=None, doc: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        '\n        Create a Command definition.\n\n        :param function:\n            a function or a lambda with the correct signature for the type of command to inject for example `def\n            mycmd(plugin, msg, args)` for a botcmd.  Note: the first parameter will be the plugin itself (equivalent to\n            self).\n        :param cmd_type:\n            defaults to `botcmd` but can be any decorator function used for errbot commands.\n        :param cmd_args: the parameters of the decorator.\n        :param cmd_kwargs: the kwargs parameter of the decorator.\n        :param name:\n            defaults to the name of the function you are passing if it is a first class function or needs to be set if\n            you use a lambda.\n        :param doc:\n            defaults to the doc of the given function if it is a first class function. It can be set for a lambda or\n            overridden for a function with this.'
        if cmd_type is None:
            from errbot import botcmd
            cmd_type = botcmd
        if name is None:
            if function.__name__ == '<lambda>':
                raise ValueError('function is a lambda (anonymous), parameter name needs to be set.')
            name = function.__name__
        self.name = name
        if cmd_kwargs is None:
            cmd_kwargs = {}
        if cmd_args is None:
            cmd_args = ()
        function.__name__ = name
        if doc:
            function.__doc__ = doc
        self.definition = cmd_type(*(function,) + cmd_args, **cmd_kwargs)

    def append_args(self, args, kwargs):
        if False:
            return 10
        from errbot import arg_botcmd, update_wrapper
        if hasattr(self.definition, '_err_command_parser'):
            update_wrapper(self.definition, args, kwargs)
        else:
            log.warning(f"Attempting to append arguments to {self.definition} isn't supported.")

class BotPluginBase(StoreMixin):
    """
    This class handle the basic needs of bot plugins like loading, unloading and creating a storage
    It is the main contract between the plugins and the bot
    """

    def __init__(self, bot, name=None):
        if False:
            i = 10
            return i + 15
        self.is_activated = False
        self.current_pollers = []
        self.current_timers = []
        self.dependencies = []
        self._dynamic_plugins = {}
        self.log = logging.getLogger(f'errbot.plugins.{name}')
        self.log.debug('Logger for plugin %s initialized...', name)
        self._bot = bot
        self.plugin_dir = bot.repo_manager.plugin_dir
        self._name = name
        super().__init__()

    @property
    def name(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Get the name of this plugin as described in its .plug file.\n\n        :return: The plugin name.\n        '
        return self._name

    @property
    def mode(self) -> str:
        if False:
            while True:
                i = 10
        "\n        Get the current active backend.\n\n        :return: the mode like 'tox', 'xmpp' etc...\n        "
        return self._bot.mode

    @property
    def bot_config(self) -> ModuleType:
        if False:
            while True:
                i = 10
        '\n        Get the bot configuration from config.py.\n        For example you can access:\n        self.bot_config.BOT_DATA_DIR\n        '
        if isinstance(self._bot.bot_config.BOT_ADMINS, str):
            self._bot.bot_config.BOT_ADMINS = (self._bot.bot_config.BOT_ADMINS,)
        return self._bot.bot_config

    @property
    def bot_identifier(self) -> Identifier:
        if False:
            i = 10
            return i + 15
        '\n        Get bot identifier on current active backend.\n\n        :return Identifier\n        '
        return self._bot.bot_identifier

    def init_storage(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        log.debug(f'Init storage for {self.name}.')
        self.open_storage(self._bot.storage_plugin, self.name)

    def activate(self) -> None:
        if False:
            print('Hello World!')
        "\n        Override if you want to do something at initialization phase (don't forget to\n        super(Gnagna, self).activate())\n        "
        self.init_storage()
        self._bot.inject_commands_from(self)
        self._bot.inject_command_filters_from(self)
        self.is_activated = True

    def deactivate(self) -> None:
        if False:
            print('Hello World!')
        "\n        Override if you want to do something at tear down phase (don't forget to super(Gnagna, self).deactivate())\n        "
        if self.current_pollers:
            log.debug('You still have active pollers at deactivation stage, I cleaned them up for you.')
            self.current_pollers = []
            for timer in self.current_timers:
                timer.cancel()
        try:
            self.close_storage()
        except StoreNotOpenError:
            pass
        self._bot.remove_command_filters_from(self)
        self._bot.remove_commands_from(self)
        self.is_activated = False
        for plugin in self._dynamic_plugins.values():
            self._bot.remove_command_filters_from(plugin)
            self._bot.remove_commands_from(plugin)

    def start_poller(self, interval: float, method: Callable[..., None], times: int=None, args: Tuple=None, kwargs: Mapping=None) -> None:
        if False:
            while True:
                i = 10
        'Starts a poller that will be called at a regular interval\n\n        :param interval: interval in seconds\n        :param method: targetted method\n        :param times:\n            number of times polling should happen (defaults to``None`` which\n            causes the polling to happen indefinitely)\n        :param args: args for the targetted method\n        :param kwargs: kwargs for the targetting method\n        '
        if not kwargs:
            kwargs = {}
        if not args:
            args = []
        log.debug(f'Programming the polling of {method.__name__} every {interval} seconds with args {str(args)} and kwargs {str(kwargs)}')
        try:
            self.current_pollers.append((method, args, kwargs))
            self.program_next_poll(interval, method, times, args, kwargs)
        except Exception:
            log.exception('Poller programming failed.')

    def stop_poller(self, method: Callable[..., None], args: Tuple=None, kwargs: Mapping=None) -> None:
        if False:
            print('Hello World!')
        if not kwargs:
            kwargs = {}
        if not args:
            args = []
        log.debug(f'Stop polling of {method} with args {args} and kwargs {kwargs}')
        self.current_pollers.remove((method, args, kwargs))

    def program_next_poll(self, interval: float, method: Callable[..., None], times: int=None, args: Tuple=None, kwargs: Mapping=None) -> None:
        if False:
            i = 10
            return i + 15
        if times is not None and times <= 0:
            return
        t = Timer(interval=interval, function=self.poller, kwargs={'interval': interval, 'method': method, 'times': times, 'args': args, 'kwargs': kwargs})
        self.current_timers.append(t)
        t.name = f'Poller thread for {type(method.__self__).__name__}'
        t.daemon = True
        t.start()

    def poller(self, interval: float, method: Callable[..., None], times: int=None, args: Tuple=None, kwargs: Mapping=None) -> None:
        if False:
            while True:
                i = 10
        previous_timer = current_thread()
        if previous_timer in self.current_timers:
            log.debug('Previous timer found and removed')
            self.current_timers.remove(previous_timer)
        if (method, args, kwargs) in self.current_pollers:
            try:
                method(*args, **kwargs)
            except Exception:
                log.exception('A poller crashed')
            if times is not None:
                times -= 1
            self.program_next_poll(interval, method, times, args, kwargs)

    def create_dynamic_plugin(self, name: str, commands: Tuple[Command], doc: str='') -> None:
        if False:
            i = 10
            return i + 15
        '\n        Creates a plugin dynamically and exposes its commands right away.\n\n        :param name: name of the plugin.\n        :param commands: a tuple of command definition.\n        :param doc: the main documentation of the plugin.\n        '
        if name in self._dynamic_plugins:
            raise ValueError('Dynamic plugin %s already created.')
        plugin_class = type(re.sub('\\W|^(?=\\d)', '_', name), (BotPlugin,), {command.name: command.definition for command in commands})
        plugin_class.__errdoc__ = doc
        plugin = plugin_class(self._bot, name=name)
        self._dynamic_plugins[name] = plugin
        self._bot.inject_commands_from(plugin)

    def destroy_dynamic_plugin(self, name: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Reverse operation of create_dynamic_plugin.\n\n        This allows you to dynamically refresh the list of commands for example.\n        :param name: the name of the dynamic plugin given to create_dynamic_plugin.\n        '
        if name not in self._dynamic_plugins:
            raise ValueError("Dynamic plugin %s doesn't exist.", name)
        plugin = self._dynamic_plugins[name]
        self._bot.remove_command_filters_from(plugin)
        self._bot.remove_commands_from(plugin)
        del self._dynamic_plugins[name]

    def get_plugin(self, name) -> 'BotPlugin':
        if False:
            while True:
                i = 10
        '\n        Gets a plugin your plugin depends on. The name of the dependency needs to be listed in [Code] section\n        key DependsOn of your plug file. This method can only be used after your plugin activation\n        (or having called super().activate() from activate itself).\n        It will return a plugin object.\n\n        :param name: the name\n        :return: the BotPlugin object requested.\n        '
        if not self.is_activated:
            raise Exception('Plugin needs to be in activated state to be able to get its dependencies.')
        if name not in self.dependencies:
            raise Exception(f'Plugin dependency {name} needs to be listed in section [Core] key "DependsOn" to be used in get_plugin.')
        return self._bot.plugin_manager.get_plugin_obj_by_name(name)

class BotPlugin(BotPluginBase):

    def get_configuration_template(self) -> Mapping:
        if False:
            print('Hello World!')
        "\n        If your plugin needs a configuration, override this method and return\n        a configuration template.\n\n        For example a dictionary like:\n        return {'LOGIN' : 'example@example.com', 'PASSWORD' : 'password'}\n\n        Note: if this method returns None, the plugin won't be configured\n        "
        return None

    def check_configuration(self, configuration: Mapping) -> None:
        if False:
            return 10
        '\n        By default, this method will do only a BASIC check. You need to override\n        it if you want to do more complex checks. It will be called before the\n        configure callback. Note if the config_template is None, it will never\n        be called.\n\n        It means recusively:\n\n        1. in case of a dictionary, it will check if all the entries and from\n           the same type are there and not more.\n        2. in case of an array or tuple, it will assume array members of the\n           same type of first element of the template (no mix typed is supported)\n\n        In case of validation error it should raise a errbot.ValidationException\n\n        :param configuration: the configuration to be checked.\n        '
        recurse_check_structure(self.get_configuration_template(), configuration)

    def configure(self, configuration: Mapping) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        By default, it will just store the current configuration in the self.config\n        field of your plugin. If this plugin has no configuration yet, the framework\n        will call this function anyway with None.\n\n        This method will be called before activation so don't expect to be activated\n        at that point.\n\n        :param configuration: injected configuration for the plugin.\n        "
        self.config = configuration

    def activate(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Triggered on plugin activation.\n\n        Override this method if you want to do something at initialization phase\n        (don't forget to `super().activate()`).\n        "
        super().activate()

    def deactivate(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Triggered on plugin deactivation.\n\n        Override this method if you want to do something at tear-down phase\n        (don't forget to `super().deactivate()`).\n        "
        super().deactivate()

    def callback_connect(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Triggered when the bot has successfully connected to the chat network.\n\n        Override this method to get notified when the bot is connected.\n        '
        pass

    def callback_message(self, message: Message) -> None:
        if False:
            print('Hello World!')
        '\n        Triggered on every message not coming from the bot itself.\n\n        Override this method to get notified on *ANY* message.\n\n        :param message:\n            representing the message that was received.\n        '
        pass

    def callback_mention(self, message: Message, mentioned_people: Sequence[Identifier]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Triggered if there are mentioned people in message.\n\n        Override this method to get notified when someone was mentioned in message.\n        [Note: This might not be implemented by all backends.]\n\n        :param message:\n            representing the message that was received.\n        :param mentioned_people:\n            all mentioned people in this message.\n        '
        pass

    def callback_presence(self, presence: Presence) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Triggered on every presence change.\n\n        :param presence:\n            An instance of :class:`~errbot.backends.base.Presence`\n            representing the new presence state that was received.\n        '
        pass

    def callback_reaction(self, reaction: Reaction) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Triggered on every reaction event.\n\n        :param reaction:\n            An instance of :class:`~errbot.backends.base.Reaction`\n            representing the new reaction event that was received.\n        '
        pass

    def callback_stream(self, stream: Stream) -> None:
        if False:
            return 10
        '\n        Triggered asynchronously (in a different thread context) on every incoming stream\n        request or file transfer request.\n        You can block this call until you are done with the stream.\n        To signal that you accept / reject the file, simply call stream.accept()\n        or stream.reject() and return.\n\n        :param stream:\n            the incoming stream request.\n        '
        stream.reject()

    def callback_botmessage(self, message: Message) -> None:
        if False:
            return 10
        '\n        Triggered on every message coming from the bot itself.\n\n        Override this method to get notified on all messages coming from\n        the bot itself (including those from other plugins).\n\n        :param message:\n            An instance of :class:`~errbot.backends.base.Message`\n            representing the message that was received.\n        '
        pass

    def callback_room_joined(self, room: Room, identifier: Identifier, invited_by: Optional[Identifier]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Triggered when a user has joined a MUC.\n\n        :param room:\n            An instance of :class:`~errbot.backends.base.MUCRoom`\n            representing the room that was joined.\n        :param identifier: An instance of Identifier (Person). Defaults to bot\n        :param invited_by: An instance of Identifier (Person). Defaults to None\n        '
        pass

    def callback_room_left(self, room: Room, identifier: Identifier, kicked_by: Optional[Identifier]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Triggered when a user has left a MUC.\n\n        :param room:\n            An instance of :class:`~errbot.backends.base.MUCRoom`\n            representing the room that was left.\n        :param identifier: An instance of Identifier (Person). Defaults to bot\n        :param kicked_by: An instance of Identifier (Person). Defaults to None\n        '
        pass

    def callback_room_topic(self, room: Room) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Triggered when the topic in a MUC changes.\n\n        :param room:\n            An instance of :class:`~errbot.backends.base.MUCRoom`\n            representing the room for which the topic changed.\n        '
        pass

    def warn_admins(self, warning: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Send a warning to the administrators of the bot.\n\n        :param warning: The markdown-formatted text of the message to send.\n        '
        self._bot.warn_admins(warning)

    def send(self, identifier: Identifier, text: str, in_reply_to: Message=None, groupchat_nick_reply: bool=False) -> None:
        if False:
            while True:
                i = 10
        '\n        Send a message to a room or a user.\n\n        :param groupchat_nick_reply: if True the message will mention the user in the chatroom.\n        :param in_reply_to: the original message this message is a reply to (optional).\n                            In some backends it will start a thread.\n        :param text: markdown formatted text to send to the user.\n        :param identifier: An Identifier representing the user or room to message.\n                           Identifiers may be created with :func:`build_identifier`.\n        '
        if not isinstance(identifier, Identifier):
            raise ValueError('identifier needs to be of type Identifier, the old string behavior is not supported')
        return self._bot.send(identifier, text, in_reply_to, groupchat_nick_reply)

    def send_card(self, body: str='', to: Identifier=None, in_reply_to: Message=None, summary: str=None, title: str='', link: str=None, image: str=None, thumbnail: str=None, color: str='green', fields: Tuple[Tuple[str, str], ...]=()) -> None:
        if False:
            while True:
                i = 10
        '\n        Sends a card.\n\n        A Card is a special type of preformatted message. If it matches with a backend similar concept like on\n        Slack it will be rendered natively, otherwise it will be sent as a regular formatted message.\n\n        :param body: main text of the card in markdown.\n        :param to: the card is sent to this identifier (Room, RoomOccupant, Person...).\n        :param in_reply_to: the original message this message is a reply to (optional).\n        :param summary: (optional) One liner summary of the card, possibly collapsed to it.\n        :param title: (optional) Title possibly linking.\n        :param link: (optional) url the title link is pointing to.\n        :param image: (optional) link to the main image of the card.\n        :param thumbnail: (optional) link to an icon / thumbnail.\n        :param color: (optional) background color or color indicator.\n        :param fields: (optional) a tuple of (key, value) pairs.\n        '
        frm = in_reply_to.to if in_reply_to else self.bot_identifier
        if to is None:
            if in_reply_to is None:
                raise ValueError('Either to or in_reply_to needs to be set.')
            to = in_reply_to.frm
        self._bot.send_card(Card(body, frm, to, in_reply_to, summary, title, link, image, thumbnail, color, fields))

    def change_presence(self, status: str=ONLINE, message: str='') -> None:
        if False:
            print('Hello World!')
        '\n            Changes the presence/status of the bot.\n\n        :param status: One of the constant defined in base.py : ONLINE, OFFLINE, DND,...\n        :param message: Additional message\n        :return: None\n        '
        self._bot.change_presence(status, message)

    def send_templated(self, identifier: Identifier, template_name: str, template_parameters: Mapping, in_reply_to: Message=None, groupchat_nick_reply: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        Sends asynchronously a message to a room or a user.\n\n        Same as send but passing a template name and parameters instead of directly the markdown text.\n        :param template_parameters: arguments for the template.\n        :param template_name: name of the template to use.\n        :param groupchat_nick_reply: if True it will mention the user in the chatroom.\n        :param in_reply_to: optionally, the original message this message is the answer to.\n        :param identifier: identifier of the user or room to which you want to send a message to.\n        '
        return self._bot.send_templated(identifier=identifier, template_name=template_name, template_parameters=template_parameters, in_reply_to=in_reply_to, groupchat_nick_reply=groupchat_nick_reply)

    def build_identifier(self, txtrep: str) -> Identifier:
        if False:
            return 10
        '\n        Transform a textual representation of a user identifier to the correct\n        Identifier object you can set in Message.to and Message.frm.\n\n        :param txtrep: the textual representation of the identifier (it is backend dependent).\n        :return: a user identifier.\n        '
        return self._bot.build_identifier(txtrep)

    def send_stream_request(self, user: Identifier, fsource: IOBase, name: str=None, size: int=None, stream_type: str=None) -> Callable:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sends asynchronously a stream/file to a user.\n\n        :param user: is the identifier of the person you want to send it to.\n        :param fsource: is a file object you want to send.\n        :param name: is an optional filename for it.\n        :param size: is optional and is the espected size for it.\n        :param stream_type: is optional for the mime_type of the content.\n\n        It will return a Stream object on which you can monitor the progress of it.\n        '
        return self._bot.send_stream_request(user, fsource, name, size, stream_type)

    def rooms(self) -> Sequence[Room]:
        if False:
            for i in range(10):
                print('nop')
        '\n        The list of rooms the bot is currently in.\n        '
        return self._bot.rooms()

    def query_room(self, room: str) -> Room:
        if False:
            print('Hello World!')
        "\n        Query a room for information.\n\n        :param room:\n            The JID/identifier of the room to query for.\n        :returns:\n            An instance of :class:`~errbot.backends.base.MUCRoom`.\n        :raises:\n            :class:`~errbot.backends.base.RoomDoesNotExistError` if the room doesn't exist.\n        "
        return self._bot.query_room(room)

    def start_poller(self, interval: float, method: Callable[..., None], times: int=None, args: Tuple=None, kwargs: Mapping=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Start to poll a method at specific interval in seconds.\n\n        Note: it will call the method with the initial interval delay for\n        the first time\n\n        Also, you can program\n        for example : self.program_poller(self, 30, fetch_stuff)\n        where you have def fetch_stuff(self) in your plugin\n\n        :param interval: interval in seconds\n        :param method: targetted method\n        :param times:\n            number of times polling should happen (defaults to``None``\n            which causes the polling to happen indefinitely)\n        :param args: args for the targetted method\n        :param kwargs: kwargs for the targetting method\n\n        '
        super().start_poller(interval, method, times, args, kwargs)

    def stop_poller(self, method: Callable[..., None], args: Tuple=None, kwargs: Mapping=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        stop poller(s).\n\n        If the method equals None -> it stops all the pollers you need to\n        regive the same parameters as the original start_poller to match a\n        specific poller to stop\n\n        :param kwargs: The initial kwargs you gave to start_poller.\n        :param args: The initial args you gave to start_poller.\n        :param method: The initial method you passed to start_poller.\n\n        '
        super().stop_poller(method, args, kwargs)

class ArgParserBase:
    """
    The `ArgSplitterBase` class defines the API which is used for argument
    splitting (used by the `split_args_with` parameter on
    :func:`~errbot.decorators.botcmd`).
    """

    def parse_args(self, args: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method takes a string of un-split arguments and parses it,\n        returning a list that is the result of splitting.\n\n        If splitting fails for any reason it should return an exception\n        of some kind.\n\n        :param args: string to parse\n        '
        raise NotImplementedError()

class SeparatorArgParser(ArgParserBase):
    """
    This argument splitter splits args on a given separator, like
    :func:`str.split` does.
    """

    def __init__(self, separator: str=None, maxsplit: int=-1):
        if False:
            while True:
                i = 10
        '\n        :param separator:\n            The separator on which arguments should be split. If sep is\n            None, any whitespace string is a separator and empty strings\n            are removed from the result.\n        :param maxsplit:\n            If given, do at most this many splits.\n        '
        self.separator = separator
        self.maxsplit = maxsplit

    def parse_args(self, args: str) -> List:
        if False:
            print('Hello World!')
        return args.split(self.separator, self.maxsplit)

class ShlexArgParser(ArgParserBase):
    """
    This argument splitter splits args using posix shell quoting rules,
    like :func:`shlex.split` does.
    """

    def parse_args(self, args):
        if False:
            print('Hello World!')
        return shlex.split(args)