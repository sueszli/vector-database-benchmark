"""
commands.requires
=================
This module manages the logic of resolving command permissions and
requirements. This includes rules which override those requirements,
as well as custom checks which can be overridden, and some special
checks like bot permissions checks.
"""
import asyncio
import enum
import inspect
from collections import ChainMap
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar, Dict, List, Mapping, Optional, Tuple, TypeVar, Union
import discord
from discord.ext.commands import check
from .errors import BotMissingPermissions
from redbot.core import utils
if TYPE_CHECKING:
    from .commands import Command
    from .context import Context
    _CommandOrCoro = TypeVar('_CommandOrCoro', Callable[..., Awaitable[Any]], Command)
__all__ = ['CheckPredicate', 'GlobalPermissionModel', 'GuildPermissionModel', 'PermissionModel', 'PrivilegeLevel', 'PermState', 'Requires', 'permissions_check', 'bot_has_permissions', 'bot_in_a_guild', 'bot_can_manage_channel', 'bot_can_react', 'has_permissions', 'can_manage_channel', 'has_guild_permissions', 'is_owner', 'guildowner', 'guildowner_or_can_manage_channel', 'guildowner_or_permissions', 'admin', 'admin_or_can_manage_channel', 'admin_or_permissions', 'mod', 'mod_or_can_manage_channel', 'mod_or_permissions', 'transition_permstate_to', 'PermStateTransitions', 'PermStateAllowedStates']
_T = TypeVar('_T')
GlobalPermissionModel = Union[discord.User, discord.VoiceChannel, discord.StageChannel, discord.TextChannel, discord.ForumChannel, discord.CategoryChannel, discord.Role, discord.Guild]
GuildPermissionModel = Union[discord.Member, discord.VoiceChannel, discord.StageChannel, discord.TextChannel, discord.ForumChannel, discord.CategoryChannel, discord.Role, discord.Guild]
PermissionModel = Union[GlobalPermissionModel, GuildPermissionModel]
CheckPredicate = Callable[['Context'], Union[Optional[bool], Awaitable[Optional[bool]]]]

class PrivilegeLevel(enum.IntEnum):
    """Enumeration for special privileges."""
    NONE = enum.auto()
    'No special privilege level.'
    MOD = enum.auto()
    'User has the mod role.'
    ADMIN = enum.auto()
    'User has the admin role.'
    GUILD_OWNER = enum.auto()
    'User is the guild level.'
    BOT_OWNER = enum.auto()
    'User is a bot owner.'

    @classmethod
    async def from_ctx(cls, ctx: 'Context') -> 'PrivilegeLevel':
        """Get a command author's PrivilegeLevel based on context."""
        if await ctx.bot.is_owner(ctx.author):
            return cls.BOT_OWNER
        elif ctx.guild is None:
            return cls.NONE
        elif ctx.author == ctx.guild.owner:
            return cls.GUILD_OWNER
        guild_settings = ctx.bot._config.guild(ctx.guild)
        for snowflake in await guild_settings.admin_role():
            if ctx.author.get_role(snowflake):
                return cls.ADMIN
        for snowflake in await guild_settings.mod_role():
            if ctx.author.get_role(snowflake):
                return cls.MOD
        return cls.NONE

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'<{self.__class__.__name__}.{self.name}>'

class PermState(enum.Enum):
    """Enumeration for permission states used by rules."""
    ACTIVE_ALLOW = enum.auto()
    'This command has been actively allowed, default user checks\n    should be ignored.\n    '
    NORMAL = enum.auto()
    'No overrides have been set for this command, make determination\n    from default user checks.\n    '
    PASSIVE_ALLOW = enum.auto()
    "There exists a subcommand in the `ACTIVE_ALLOW` state, continue\n    down the subcommand tree until we either find it or realise we're\n    on the wrong branch.\n    "
    CAUTIOUS_ALLOW = enum.auto()
    'This command has been actively denied, but there exists a\n    subcommand in the `ACTIVE_ALLOW` state. This occurs when\n    `PASSIVE_ALLOW` and `ACTIVE_DENY` are combined.\n    '
    ACTIVE_DENY = enum.auto()
    'This command has been actively denied, terminate the command\n    chain.\n    '
    ALLOWED_BY_HOOK = enum.auto()
    'This command has been actively allowed by a permission hook.\n    check validation swaps this out, but the information may be useful\n    to developers. It is treated as `ACTIVE_ALLOW` for the current command\n    and `PASSIVE_ALLOW` for subcommands.'
    DENIED_BY_HOOK = enum.auto()
    'This command has been actively denied by a permission hook\n    check validation swaps this out, but the information may be useful\n    to developers. It is treated as `ACTIVE_DENY` for the current command\n    and any subcommands.'

    @classmethod
    def from_bool(cls, value: Optional[bool]) -> 'PermState':
        if False:
            return 10
        'Get a PermState from a bool or ``NoneType``.'
        if value is True:
            return cls.ACTIVE_ALLOW
        elif value is False:
            return cls.ACTIVE_DENY
        else:
            return cls.NORMAL

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<{self.__class__.__name__}.{self.name}>'
TransitionResult = Tuple[Optional[bool], Union[PermState, Dict[bool, PermState]]]
TransitionDict = Dict[PermState, Dict[PermState, TransitionResult]]
PermStateTransitions: TransitionDict = {PermState.ACTIVE_ALLOW: {PermState.ACTIVE_ALLOW: (True, PermState.ACTIVE_ALLOW), PermState.NORMAL: (True, PermState.ACTIVE_ALLOW), PermState.PASSIVE_ALLOW: (True, PermState.ACTIVE_ALLOW), PermState.CAUTIOUS_ALLOW: (True, PermState.CAUTIOUS_ALLOW), PermState.ACTIVE_DENY: (False, PermState.ACTIVE_DENY)}, PermState.NORMAL: {PermState.ACTIVE_ALLOW: (True, PermState.ACTIVE_ALLOW), PermState.NORMAL: (None, PermState.NORMAL), PermState.PASSIVE_ALLOW: (True, {True: PermState.NORMAL, False: PermState.PASSIVE_ALLOW}), PermState.CAUTIOUS_ALLOW: (True, PermState.CAUTIOUS_ALLOW), PermState.ACTIVE_DENY: (False, PermState.ACTIVE_DENY)}, PermState.PASSIVE_ALLOW: {PermState.ACTIVE_ALLOW: (True, PermState.ACTIVE_ALLOW), PermState.NORMAL: (False, PermState.NORMAL), PermState.PASSIVE_ALLOW: (True, PermState.PASSIVE_ALLOW), PermState.CAUTIOUS_ALLOW: (True, PermState.CAUTIOUS_ALLOW), PermState.ACTIVE_DENY: (False, PermState.ACTIVE_DENY)}, PermState.CAUTIOUS_ALLOW: {PermState.ACTIVE_ALLOW: (True, PermState.ACTIVE_ALLOW), PermState.NORMAL: (False, PermState.ACTIVE_DENY), PermState.PASSIVE_ALLOW: (True, PermState.CAUTIOUS_ALLOW), PermState.CAUTIOUS_ALLOW: (True, PermState.CAUTIOUS_ALLOW), PermState.ACTIVE_DENY: (False, PermState.ACTIVE_DENY)}, PermState.ACTIVE_DENY: {PermState.ACTIVE_ALLOW: (True, PermState.ACTIVE_ALLOW), PermState.NORMAL: (False, PermState.ACTIVE_DENY), PermState.PASSIVE_ALLOW: (False, PermState.ACTIVE_DENY), PermState.CAUTIOUS_ALLOW: (False, PermState.ACTIVE_DENY), PermState.ACTIVE_DENY: (False, PermState.ACTIVE_DENY)}}
PermStateAllowedStates = (PermState.ACTIVE_ALLOW, PermState.PASSIVE_ALLOW, PermState.CAUTIOUS_ALLOW)

def transition_permstate_to(prev: PermState, next_state: PermState) -> TransitionResult:
    if False:
        i = 10
        return i + 15
    if prev is PermState.ALLOWED_BY_HOOK:
        prev = PermState.PASSIVE_ALLOW
    elif prev is PermState.DENIED_BY_HOOK:
        prev = PermState.ACTIVE_DENY
    return PermStateTransitions[prev][next_state]

class Requires:
    """This class describes the requirements for executing a specific command.

    The permissions described include both bot permissions and user
    permissions.

    Attributes
    ----------
    checks : List[Callable[[Context], Union[bool, Awaitable[bool]]]]
        A list of checks which can be overridden by rules. Use
        `Command.checks` if you would like them to never be overridden.
    privilege_level : PrivilegeLevel
        The required privilege level (bot owner, admin, etc.) for users
        to execute the command. Can be ``None``, in which case the
        `user_perms` will be used exclusively, otherwise, for levels
        other than bot owner, the user can still run the command if
        they have the required `user_perms`.
    ready_event : asyncio.Event
        Event for when this Requires object has had its rules loaded.
        If permissions is loaded, this should be set when permissions
        has finished loading rules into this object. If permissions
        is not loaded, it should be set as soon as the command or cog
        is added.
    user_perms : Optional[discord.Permissions]
        The required permissions for users to execute the command. Can
        be ``None``, in which case the `privilege_level` will be used
        exclusively, otherwise, it will pass whether the user has the
        required `privilege_level` _or_ `user_perms`.
    bot_perms : discord.Permissions
        The required bot permissions for a command to be executed. This
        is not overrideable by other conditions.

    """
    DEFAULT: ClassVar[str] = 'default'
    'The key for the default rule in a rules dict.'
    GLOBAL: ClassVar[int] = 0
    'Should be used in place of a guild ID when setting/getting\n    global rules.\n    '

    def __init__(self, privilege_level: Optional[PrivilegeLevel], user_perms: Union[Dict[str, bool], discord.Permissions, None], bot_perms: Union[Dict[str, bool], discord.Permissions], checks: List[CheckPredicate]):
        if False:
            print('Hello World!')
        self.checks: List[CheckPredicate] = checks
        self.privilege_level: Optional[PrivilegeLevel] = privilege_level
        self.ready_event = asyncio.Event()
        if isinstance(user_perms, dict):
            self.user_perms: Optional[discord.Permissions] = discord.Permissions.none()
            _validate_perms_dict(user_perms)
            self.user_perms.update(**user_perms)
        else:
            self.user_perms = user_perms
        if isinstance(bot_perms, dict):
            self.bot_perms: discord.Permissions = discord.Permissions.none()
            _validate_perms_dict(bot_perms)
            self.bot_perms.update(**bot_perms)
        else:
            self.bot_perms = bot_perms
        self._global_rules: _RulesDict = _RulesDict()
        self._guild_rules: _IntKeyDict[_RulesDict] = _IntKeyDict[_RulesDict]()

    @staticmethod
    def get_decorator(privilege_level: Optional[PrivilegeLevel], user_perms: Optional[Dict[str, bool]]) -> Callable[['_CommandOrCoro'], '_CommandOrCoro']:
        if False:
            while True:
                i = 10
        if not user_perms:
            user_perms = None

        def decorator(func: '_CommandOrCoro') -> '_CommandOrCoro':
            if False:
                i = 10
                return i + 15
            if inspect.iscoroutinefunction(func):
                func.__requires_privilege_level__ = privilege_level
                if user_perms is None:
                    func.__requires_user_perms__ = None
                else:
                    if getattr(func, '__requires_user_perms__', None) is None:
                        func.__requires_user_perms__ = discord.Permissions.none()
                    func.__requires_user_perms__.update(**user_perms)
            else:
                func.requires.privilege_level = privilege_level
                if user_perms is None:
                    func.requires.user_perms = None
                else:
                    _validate_perms_dict(user_perms)
                    if func.requires.user_perms is None:
                        func.requires.user_perms = discord.Permissions.none()
                    func.requires.user_perms.update(**user_perms)
            return func
        return decorator

    def get_rule(self, model: Union[int, str, PermissionModel], guild_id: int) -> PermState:
        if False:
            while True:
                i = 10
        "Get the rule for a particular model.\n\n        Parameters\n        ----------\n        model : Union[int, str, PermissionModel]\n            The model to get the rule for. `str` is only valid for\n            `Requires.DEFAULT`.\n        guild_id : int\n            The ID of the guild for the rule's scope. Set to\n            `Requires.GLOBAL` for a global rule.\n            If a global rule is set for a model,\n            it will be preferred over the guild rule.\n\n        Returns\n        -------\n        PermState\n            The state for this rule. See the `PermState` class\n            for an explanation.\n\n        "
        if not isinstance(model, (str, int)):
            model = model.id
        rules: Mapping[Union[int, str], PermState]
        if guild_id:
            rules = ChainMap(self._global_rules, self._guild_rules.get(guild_id, _RulesDict()))
        else:
            rules = self._global_rules
        return rules.get(model, PermState.NORMAL)

    def set_rule(self, model_id: Union[str, int], rule: PermState, guild_id: int) -> None:
        if False:
            i = 10
            return i + 15
        "Set the rule for a particular model.\n\n        Parameters\n        ----------\n        model_id : Union[str, int]\n            The model to add a rule for. `str` is only valid for\n            `Requires.DEFAULT`.\n        rule : PermState\n            Which state this rule should be set as. See the `PermState`\n            class for an explanation.\n        guild_id : int\n            The ID of the guild for the rule's scope. Set to\n            `Requires.GLOBAL` for a global rule.\n\n        "
        if guild_id:
            rules = self._guild_rules.setdefault(guild_id, _RulesDict())
        else:
            rules = self._global_rules
        if rule is PermState.NORMAL:
            rules.pop(model_id, None)
        else:
            rules[model_id] = rule

    def clear_all_rules(self, guild_id: int, *, preserve_default_rule: bool=True) -> None:
        if False:
            print('Hello World!')
        'Clear all rules of a particular scope.\n\n        Parameters\n        ----------\n        guild_id : int\n            The guild ID to clear rules for. If set to\n            `Requires.GLOBAL`, this will clear all global rules and\n            leave all guild rules untouched.\n\n        Other Parameters\n        ----------------\n        preserve_default_rule : bool\n            Whether to preserve the default rule or not.\n            This defaults to being preserved\n\n        '
        if guild_id:
            rules = self._guild_rules.setdefault(guild_id, _RulesDict())
        else:
            rules = self._global_rules
        default = rules.get(self.DEFAULT, None)
        rules.clear()
        if default is not None and preserve_default_rule:
            rules[self.DEFAULT] = default

    def reset(self) -> None:
        if False:
            return 10
        'Reset this Requires object to its original state.\n\n        This will clear all rules, including defaults. It also resets\n        the `Requires.ready_event`.\n        '
        self._guild_rules.clear()
        self._global_rules.clear()
        self.ready_event.clear()

    async def verify(self, ctx: 'Context') -> bool:
        """Check if the given context passes the requirements.

        This will check the bot permissions, overrides, user permissions
        and privilege level.

        Parameters
        ----------
        ctx : "Context"
            The invocation context to check with.

        Returns
        -------
        bool
            ``True`` if the context passes the requirements.

        Raises
        ------
        BotMissingPermissions
            If the bot is missing required permissions to run the
            command.
        CommandError
            Propagated from any permissions checks.

        """
        if not self.ready_event.is_set():
            await self.ready_event.wait()
        await self._verify_bot(ctx)
        if await ctx.bot.is_owner(ctx.author):
            return True
        if self.privilege_level is PrivilegeLevel.BOT_OWNER:
            return False
        hook_result = await ctx.bot.verify_permissions_hooks(ctx)
        if hook_result is not None:
            return hook_result
        return await self._transition_state(ctx)

    async def _verify_bot(self, ctx: 'Context') -> None:
        cog = ctx.cog
        if ctx.guild is not None and cog and await ctx.bot.cog_disabled_in_guild(cog, ctx.guild):
            raise discord.ext.commands.DisabledCommand()
        bot_perms = ctx.bot_permissions
        if not (bot_perms.administrator or bot_perms >= self.bot_perms):
            raise BotMissingPermissions(missing=self._missing_perms(self.bot_perms, bot_perms))

    async def _transition_state(self, ctx: 'Context') -> bool:
        (should_invoke, next_state) = self._get_transitioned_state(ctx)
        if should_invoke is None:
            should_invoke = await self._verify_user(ctx)
        elif isinstance(next_state, dict):
            would_invoke = self._get_would_invoke(ctx)
            if would_invoke is None:
                would_invoke = await self._verify_user(ctx)
            next_state = next_state[would_invoke]
        assert isinstance(next_state, PermState)
        ctx.permission_state = next_state
        return should_invoke

    def _get_transitioned_state(self, ctx: 'Context') -> TransitionResult:
        if False:
            for i in range(10):
                print('nop')
        prev_state = ctx.permission_state
        cur_state = self._get_rule_from_ctx(ctx)
        return transition_permstate_to(prev_state, cur_state)

    def _get_would_invoke(self, ctx: 'Context') -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        default_rule = PermState.NORMAL
        if ctx.guild is not None:
            default_rule = self.get_rule(self.DEFAULT, guild_id=ctx.guild.id)
        if default_rule is PermState.NORMAL:
            default_rule = self.get_rule(self.DEFAULT, self.GLOBAL)
        if default_rule == PermState.ACTIVE_DENY:
            return False
        elif default_rule == PermState.ACTIVE_ALLOW:
            return True
        else:
            return None

    async def _verify_user(self, ctx: 'Context') -> bool:
        checks_pass = await self._verify_checks(ctx)
        if checks_pass is False:
            return False
        if self.user_perms is not None:
            user_perms = ctx.permissions
            if user_perms.administrator or user_perms >= self.user_perms:
                return True
        if self.privilege_level is not None:
            privilege_level = await PrivilegeLevel.from_ctx(ctx)
            if privilege_level >= self.privilege_level:
                return True
        return False

    def _get_rule_from_ctx(self, ctx: 'Context') -> PermState:
        if False:
            i = 10
            return i + 15
        author = ctx.author
        guild = ctx.guild
        if ctx.guild is None:
            rule = self._global_rules.get(author.id)
            if rule is not None:
                return rule
            return self.get_rule(self.DEFAULT, self.GLOBAL)
        rules_chain = [self._global_rules]
        guild_rules = self._guild_rules.get(ctx.guild.id)
        if guild_rules:
            rules_chain.append(guild_rules)
        channels = []
        if author.voice is not None:
            channels.append(author.voice.channel)
        if isinstance(ctx.channel, discord.Thread):
            channels.append(ctx.channel.parent)
        else:
            channels.append(ctx.channel)
        category = ctx.channel.category
        if category is not None:
            channels.append(category)
        author_roles = reversed(author.roles[1:])
        model_chain = [author, *channels, *author_roles, guild]
        for rules in rules_chain:
            for model in model_chain:
                rule = rules.get(model.id)
                if rule is not None:
                    return rule
            del model_chain[-1]
        default_rule = self.get_rule(self.DEFAULT, guild.id)
        if default_rule is PermState.NORMAL:
            default_rule = self.get_rule(self.DEFAULT, self.GLOBAL)
        return default_rule

    async def _verify_checks(self, ctx: 'Context') -> bool:
        if not self.checks:
            return True
        return await discord.utils.async_all((check(ctx) for check in self.checks))

    @staticmethod
    def _missing_perms(required: discord.Permissions, actual: discord.Permissions) -> discord.Permissions:
        if False:
            i = 10
            return i + 15
        relative_complement = required.value & ~actual.value
        return discord.Permissions(relative_complement)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<Requires privilege_level={self.privilege_level!r} user_perms={self.user_perms!r} bot_perms={self.bot_perms!r}>'

def permissions_check(predicate: CheckPredicate):
    if False:
        i = 10
        return i + 15
    'An overwriteable version of `discord.ext.commands.check`.\n\n    This has the same behaviour as `discord.ext.commands.check`,\n    however this check can be ignored if the command is allowed\n    through a permissions cog.\n    '

    def decorator(func: '_CommandOrCoro') -> '_CommandOrCoro':
        if False:
            for i in range(10):
                print('nop')
        if hasattr(func, 'requires'):
            func.requires.checks.append(predicate)
        else:
            if not hasattr(func, '__requires_checks__'):
                func.__requires_checks__ = []
            func.__requires_checks__.append(predicate)
        return func
    return decorator

def has_guild_permissions(**perms):
    if False:
        return 10
    'Restrict the command to users with these guild permissions.\n\n    This check can be overridden by rules.\n    '
    _validate_perms_dict(perms)

    def predicate(ctx):
        if False:
            for i in range(10):
                print('nop')
        return ctx.guild and ctx.author.guild_permissions >= discord.Permissions(**perms)
    return permissions_check(predicate)

def bot_has_permissions(**perms: bool):
    if False:
        for i in range(10):
            print('nop')
    'Complain if the bot is missing permissions.\n\n    If the user tries to run the command, but the bot is missing the\n    permissions, it will send a message describing which permissions\n    are missing.\n\n    This check cannot be overridden by rules.\n    '

    def decorator(func: '_CommandOrCoro') -> '_CommandOrCoro':
        if False:
            print('Hello World!')
        if asyncio.iscoroutinefunction(func):
            if not hasattr(func, '__requires_bot_perms__'):
                func.__requires_bot_perms__ = discord.Permissions.none()
            _validate_perms_dict(perms)
            func.__requires_bot_perms__.update(**perms)
        else:
            _validate_perms_dict(perms)
            func.requires.bot_perms.update(**perms)
        return func
    return decorator

def bot_in_a_guild():
    if False:
        for i in range(10):
            print('nop')
    'Deny the command if the bot is not in a guild.'

    async def predicate(ctx):
        return len(ctx.bot.guilds) > 0
    return check(predicate)

def bot_can_manage_channel(*, allow_thread_owner: bool=False) -> Callable[[_T], _T]:
    if False:
        while True:
            i = 10
    "\n    Complain if the bot is missing permissions to manage channel.\n\n    This check properly resolves the permissions for `discord.Thread` as well.\n\n    Parameters\n    ----------\n    allow_thread_owner: bool\n        If ``True``, the command will also be allowed to run if the bot is a thread owner.\n        This can, for example, be useful to check if the bot can edit a channel/thread's name\n        as that, in addition to members with manage channel/threads permission,\n        can also be done by the thread owner.\n    "

    def predicate(ctx: 'Context') -> bool:
        if False:
            for i in range(10):
                print('nop')
        if ctx.guild is None:
            return False
        if not utils.can_user_manage_channel(ctx.me, ctx.channel, allow_thread_owner=allow_thread_owner):
            if isinstance(ctx.channel, discord.Thread):
                missing = discord.Permissions(manage_threads=True)
            else:
                missing = discord.Permissions(manage_channels=True)
            raise BotMissingPermissions(missing=missing)
        return True
    return check(predicate)

def bot_can_react() -> Callable[[_T], _T]:
    if False:
        i = 10
        return i + 15
    '\n    Complain if the bot is missing permissions to react.\n\n    This check properly resolves the permissions for `discord.Thread` as well.\n    '

    async def predicate(ctx: 'Context') -> bool:
        return not (isinstance(ctx.channel, discord.Thread) and ctx.channel.archived)

    def decorator(func: _T) -> _T:
        if False:
            while True:
                i = 10
        func = bot_has_permissions(read_message_history=True, add_reactions=True)(func)
        func = check(predicate)(func)
        return func
    return decorator

def _can_manage_channel_deco(privilege_level: Optional[PrivilegeLevel]=None, allow_thread_owner: bool=False) -> Callable[[_T], _T]:
    if False:
        return 10

    async def predicate(ctx: 'Context') -> bool:
        if utils.can_user_manage_channel(ctx.author, ctx.channel, allow_thread_owner=allow_thread_owner):
            return True
        if privilege_level is not None:
            if await PrivilegeLevel.from_ctx(ctx) >= privilege_level:
                return True
        return False
    return permissions_check(predicate)

def has_permissions(**perms: bool):
    if False:
        return 10
    'Restrict the command to users with these permissions.\n\n    This check can be overridden by rules.\n    '
    if perms is None:
        raise TypeError('Must provide at least one keyword argument to has_permissions')
    return Requires.get_decorator(None, perms)

def can_manage_channel(*, allow_thread_owner: bool=False) -> Callable[[_T], _T]:
    if False:
        print('Hello World!')
    "Restrict the command to users with permissions to manage channel.\n\n    This check properly resolves the permissions for `discord.Thread` as well.\n\n    This check can be overridden by rules.\n\n    Parameters\n    ----------\n    allow_thread_owner: bool\n        If ``True``, the command will also be allowed to run if the author is a thread owner.\n        This can, for example, be useful to check if the author can edit a channel/thread's name\n        as that, in addition to members with manage channel/threads permission,\n        can also be done by the thread owner.\n    "
    return _can_manage_channel_deco(allow_thread_owner)

def is_owner():
    if False:
        print('Hello World!')
    'Restrict the command to bot owners.\n\n    This check cannot be overridden by rules.\n    '
    return Requires.get_decorator(PrivilegeLevel.BOT_OWNER, {})

def guildowner_or_permissions(**perms: bool):
    if False:
        while True:
            i = 10
    'Restrict the command to the guild owner or users with these permissions.\n\n    This check can be overridden by rules.\n    '
    return Requires.get_decorator(PrivilegeLevel.GUILD_OWNER, perms)

def guildowner_or_can_manage_channel(*, allow_thread_owner: bool=False) -> Callable[[_T], _T]:
    if False:
        while True:
            i = 10
    "Restrict the command to the guild owner or user with permissions to manage channel.\n\n    This check properly resolves the permissions for `discord.Thread` as well.\n\n    This check can be overridden by rules.\n\n    Parameters\n    ----------\n    allow_thread_owner: bool\n        If ``True``, the command will also be allowed to run if the author is a thread owner.\n        This can, for example, be useful to check if the author can edit a channel/thread's name\n        as that, in addition to members with manage channel/threads permission,\n        can also be done by the thread owner.\n    "
    return _can_manage_channel_deco(PrivilegeLevel.GUILD_OWNER, allow_thread_owner)

def guildowner():
    if False:
        i = 10
        return i + 15
    'Restrict the command to the guild owner.\n\n    This check can be overridden by rules.\n    '
    return guildowner_or_permissions()

def admin_or_permissions(**perms: bool):
    if False:
        for i in range(10):
            print('nop')
    'Restrict the command to users with the admin role or these permissions.\n\n    This check can be overridden by rules.\n    '
    return Requires.get_decorator(PrivilegeLevel.ADMIN, perms)

def admin_or_can_manage_channel(*, allow_thread_owner: bool=False) -> Callable[[_T], _T]:
    if False:
        for i in range(10):
            print('nop')
    "Restrict the command to users with the admin role or permissions to manage channel.\n\n    This check properly resolves the permissions for `discord.Thread` as well.\n\n    This check can be overridden by rules.\n\n    Parameters\n    ----------\n    allow_thread_owner: bool\n        If ``True``, the command will also be allowed to run if the author is a thread owner.\n        This can, for example, be useful to check if the author can edit a channel/thread's name\n        as that, in addition to members with manage channel/threads permission,\n        can also be done by the thread owner.\n    "
    return _can_manage_channel_deco(PrivilegeLevel.ADMIN, allow_thread_owner)

def admin():
    if False:
        print('Hello World!')
    'Restrict the command to users with the admin role.\n\n    This check can be overridden by rules.\n    '
    return admin_or_permissions()

def mod_or_permissions(**perms: bool):
    if False:
        while True:
            i = 10
    'Restrict the command to users with the mod role or these permissions.\n\n    This check can be overridden by rules.\n    '
    return Requires.get_decorator(PrivilegeLevel.MOD, perms)

def mod_or_can_manage_channel(*, allow_thread_owner: bool=False) -> Callable[[_T], _T]:
    if False:
        i = 10
        return i + 15
    "Restrict the command to users with the mod role or permissions to manage channel.\n\n    This check properly resolves the permissions for `discord.Thread` as well.\n\n    This check can be overridden by rules.\n\n    Parameters\n    ----------\n    allow_thread_owner: bool\n        If ``True``, the command will also be allowed to run if the author is a thread owner.\n        This can, for example, be useful to check if the author can edit a channel/thread's name\n        as that, in addition to members with manage channel/threads permission,\n        can also be done by the thread owner.\n    "
    return _can_manage_channel_deco(PrivilegeLevel.MOD, allow_thread_owner)

def mod():
    if False:
        while True:
            i = 10
    'Restrict the command to users with the mod role.\n\n    This check can be overridden by rules.\n    '
    return mod_or_permissions()

class _IntKeyDict(Dict[int, _T]):
    """Dict subclass which throws TypeError when a non-int key is used."""
    get: Callable
    setdefault: Callable

    def __getitem__(self, key: Any) -> _T:
        if False:
            i = 10
            return i + 15
        if not isinstance(key, int):
            raise TypeError('Keys must be of type `int`')
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: _T) -> None:
        if False:
            return 10
        if not isinstance(key, int):
            raise TypeError('Keys must be of type `int`')
        return super().__setitem__(key, value)

class _RulesDict(Dict[Union[int, str], PermState]):
    """Dict subclass which throws a TypeError when an invalid key is used."""
    get: Callable
    setdefault: Callable

    def __getitem__(self, key: Any) -> PermState:
        if False:
            while True:
                i = 10
        if key != Requires.DEFAULT and (not isinstance(key, int)):
            raise TypeError(f'Expected "{Requires.DEFAULT}" or int key, not "{key}"')
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: PermState) -> None:
        if False:
            print('Hello World!')
        if key != Requires.DEFAULT and (not isinstance(key, int)):
            raise TypeError(f'Expected "{Requires.DEFAULT}" or int key, not "{key}"')
        return super().__setitem__(key, value)

def _validate_perms_dict(perms: Dict[str, bool]) -> None:
    if False:
        for i in range(10):
            print('nop')
    invalid_keys = set(perms.keys()) - set(discord.Permissions.VALID_FLAGS)
    if invalid_keys:
        raise TypeError(f"Invalid perm name(s): {', '.join(invalid_keys)}")
    for (perm, value) in perms.items():
        if value is not True:
            raise TypeError(f"Permission {perm} may only be specified as 'True', not {value}")