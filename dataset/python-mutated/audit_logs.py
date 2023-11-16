"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping, Generator, List, Optional, Tuple, Type, TypeVar, Union
from . import enums, flags, utils
from .asset import Asset
from .colour import Colour
from .invite import Invite
from .mixins import Hashable
from .object import Object
from .permissions import PermissionOverwrite, Permissions
from .automod import AutoModTrigger, AutoModRuleAction, AutoModRule
from .role import Role
from .emoji import Emoji
from .partial_emoji import PartialEmoji
from .member import Member
from .scheduled_event import ScheduledEvent
from .stage_instance import StageInstance
from .sticker import GuildSticker
from .threads import Thread
from .integrations import PartialIntegration
from .channel import ForumChannel, StageChannel, ForumTag
__all__ = ('AuditLogDiff', 'AuditLogChanges', 'AuditLogEntry')
if TYPE_CHECKING:
    import datetime
    from . import abc
    from .guild import Guild
    from .state import ConnectionState
    from .types.audit_log import AuditLogChange as AuditLogChangePayload, AuditLogEntry as AuditLogEntryPayload, _AuditLogChange_TriggerMetadata as AuditLogChangeTriggerMetadataPayload
    from .types.channel import PermissionOverwrite as PermissionOverwritePayload, ForumTag as ForumTagPayload, DefaultReaction as DefaultReactionPayload
    from .types.invite import Invite as InvitePayload
    from .types.role import Role as RolePayload
    from .types.snowflake import Snowflake
    from .types.command import ApplicationCommandPermissions
    from .types.automod import AutoModerationAction
    from .user import User
    from .app_commands import AppCommand
    from .webhook import Webhook
    TargetType = Union[Guild, abc.GuildChannel, Member, User, Role, Invite, Emoji, StageInstance, GuildSticker, Thread, Object, PartialIntegration, AutoModRule, ScheduledEvent, Webhook, AppCommand, None]

def _transform_timestamp(entry: AuditLogEntry, data: Optional[str]) -> Optional[datetime.datetime]:
    if False:
        print('Hello World!')
    return utils.parse_time(data)

def _transform_color(entry: AuditLogEntry, data: int) -> Colour:
    if False:
        i = 10
        return i + 15
    return Colour(data)

def _transform_snowflake(entry: AuditLogEntry, data: Snowflake) -> int:
    if False:
        while True:
            i = 10
    return int(data)

def _transform_channel(entry: AuditLogEntry, data: Optional[Snowflake]) -> Optional[Union[abc.GuildChannel, Object]]:
    if False:
        i = 10
        return i + 15
    if data is None:
        return None
    return entry.guild.get_channel(int(data)) or Object(id=data)

def _transform_channels_or_threads(entry: AuditLogEntry, data: List[Snowflake]) -> List[Union[abc.GuildChannel, Thread, Object]]:
    if False:
        while True:
            i = 10
    return [entry.guild.get_channel_or_thread(int(data)) or Object(id=data) for data in data]

def _transform_member_id(entry: AuditLogEntry, data: Optional[Snowflake]) -> Union[Member, User, None]:
    if False:
        while True:
            i = 10
    if data is None:
        return None
    return entry._get_member(int(data))

def _transform_guild_id(entry: AuditLogEntry, data: Optional[Snowflake]) -> Optional[Guild]:
    if False:
        for i in range(10):
            print('nop')
    if data is None:
        return None
    return entry._state._get_guild(int(data))

def _transform_roles(entry: AuditLogEntry, data: List[Snowflake]) -> List[Union[Role, Object]]:
    if False:
        while True:
            i = 10
    return [entry.guild.get_role(int(role_id)) or Object(role_id, type=Role) for role_id in data]

def _transform_applied_forum_tags(entry: AuditLogEntry, data: List[Snowflake]) -> List[Union[ForumTag, Object]]:
    if False:
        while True:
            i = 10
    thread = entry.target
    if isinstance(thread, Thread) and isinstance(thread.parent, ForumChannel):
        return [thread.parent.get_tag(tag_id) or Object(id=tag_id, type=ForumTag) for tag_id in map(int, data)]
    return [Object(id=tag_id, type=ForumTag) for tag_id in data]

def _transform_overloaded_flags(entry: AuditLogEntry, data: int) -> Union[int, flags.ChannelFlags]:
    if False:
        while True:
            i = 10
    channel_audit_log_types = (enums.AuditLogAction.channel_create, enums.AuditLogAction.channel_update, enums.AuditLogAction.channel_delete, enums.AuditLogAction.thread_create, enums.AuditLogAction.thread_update, enums.AuditLogAction.thread_delete)
    if entry.action in channel_audit_log_types:
        return flags.ChannelFlags._from_value(data)
    return data

def _transform_forum_tags(entry: AuditLogEntry, data: List[ForumTagPayload]) -> List[ForumTag]:
    if False:
        while True:
            i = 10
    return [ForumTag.from_data(state=entry._state, data=d) for d in data]

def _transform_default_reaction(entry: AuditLogEntry, data: DefaultReactionPayload) -> Optional[PartialEmoji]:
    if False:
        while True:
            i = 10
    if data is None:
        return None
    emoji_name = data.get('emoji_name') or ''
    emoji_id = utils._get_as_snowflake(data, 'emoji_id') or None
    return PartialEmoji.with_state(state=entry._state, name=emoji_name, id=emoji_id)

def _transform_overwrites(entry: AuditLogEntry, data: List[PermissionOverwritePayload]) -> List[Tuple[Object, PermissionOverwrite]]:
    if False:
        for i in range(10):
            print('nop')
    overwrites = []
    for elem in data:
        allow = Permissions(int(elem['allow']))
        deny = Permissions(int(elem['deny']))
        ow = PermissionOverwrite.from_pair(allow, deny)
        ow_type = elem['type']
        ow_id = int(elem['id'])
        target = None
        if ow_type == '0':
            target = entry.guild.get_role(ow_id)
        elif ow_type == '1':
            target = entry._get_member(ow_id)
        if target is None:
            target = Object(id=ow_id, type=Role if ow_type == '0' else Member)
        overwrites.append((target, ow))
    return overwrites

def _transform_icon(entry: AuditLogEntry, data: Optional[str]) -> Optional[Asset]:
    if False:
        return 10
    if data is None:
        return None
    if entry.action is enums.AuditLogAction.guild_update:
        return Asset._from_guild_icon(entry._state, entry.guild.id, data)
    else:
        return Asset._from_icon(entry._state, entry._target_id, data, path='role')

def _transform_avatar(entry: AuditLogEntry, data: Optional[str]) -> Optional[Asset]:
    if False:
        while True:
            i = 10
    if data is None:
        return None
    return Asset._from_avatar(entry._state, entry._target_id, data)

def _transform_cover_image(entry: AuditLogEntry, data: Optional[str]) -> Optional[Asset]:
    if False:
        i = 10
        return i + 15
    if data is None:
        return None
    return Asset._from_scheduled_event_cover_image(entry._state, entry._target_id, data)

def _guild_hash_transformer(path: str) -> Callable[[AuditLogEntry, Optional[str]], Optional[Asset]]:
    if False:
        print('Hello World!')

    def _transform(entry: AuditLogEntry, data: Optional[str]) -> Optional[Asset]:
        if False:
            return 10
        if data is None:
            return None
        return Asset._from_guild_image(entry._state, entry.guild.id, data, path=path)
    return _transform

def _transform_automod_actions(entry: AuditLogEntry, data: List[AutoModerationAction]) -> List[AutoModRuleAction]:
    if False:
        print('Hello World!')
    return [AutoModRuleAction.from_data(action) for action in data]
E = TypeVar('E', bound=enums.Enum)

def _enum_transformer(enum: Type[E]) -> Callable[[AuditLogEntry, int], E]:
    if False:
        print('Hello World!')

    def _transform(entry: AuditLogEntry, data: int) -> E:
        if False:
            return 10
        return enums.try_enum(enum, data)
    return _transform
F = TypeVar('F', bound=flags.BaseFlags)

def _flag_transformer(cls: Type[F]) -> Callable[[AuditLogEntry, Union[int, str]], F]:
    if False:
        for i in range(10):
            print('nop')

    def _transform(entry: AuditLogEntry, data: Union[int, str]) -> F:
        if False:
            for i in range(10):
                print('nop')
        return cls._from_value(int(data))
    return _transform

def _transform_type(entry: AuditLogEntry, data: Union[int, str]) -> Union[enums.ChannelType, enums.StickerType, enums.WebhookType, str]:
    if False:
        print('Hello World!')
    if entry.action.name.startswith('sticker_'):
        return enums.try_enum(enums.StickerType, data)
    elif entry.action.name.startswith('integration_'):
        return data
    elif entry.action.name.startswith('webhook_'):
        return enums.try_enum(enums.WebhookType, data)
    else:
        return enums.try_enum(enums.ChannelType, data)

class AuditLogDiff:

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self.__dict__)

    def __iter__(self) -> Generator[Tuple[str, Any], None, None]:
        if False:
            i = 10
            return i + 15
        yield from self.__dict__.items()

    def __repr__(self) -> str:
        if False:
            return 10
        values = ' '.join(('%s=%r' % item for item in self.__dict__.items()))
        return f'<AuditLogDiff {values}>'
    if TYPE_CHECKING:

        def __getattr__(self, item: str) -> Any:
            if False:
                i = 10
                return i + 15
            ...

        def __setattr__(self, key: str, value: Any) -> Any:
            if False:
                i = 10
                return i + 15
            ...
Transformer = Callable[['AuditLogEntry', Any], Any]

class AuditLogChanges:
    TRANSFORMERS: ClassVar[Mapping[str, Tuple[Optional[str], Optional[Transformer]]]] = {'verification_level': (None, _enum_transformer(enums.VerificationLevel)), 'explicit_content_filter': (None, _enum_transformer(enums.ContentFilter)), 'allow': (None, _flag_transformer(Permissions)), 'deny': (None, _flag_transformer(Permissions)), 'permissions': (None, _flag_transformer(Permissions)), 'id': (None, _transform_snowflake), 'color': ('colour', _transform_color), 'owner_id': ('owner', _transform_member_id), 'inviter_id': ('inviter', _transform_member_id), 'channel_id': ('channel', _transform_channel), 'afk_channel_id': ('afk_channel', _transform_channel), 'system_channel_id': ('system_channel', _transform_channel), 'system_channel_flags': (None, _flag_transformer(flags.SystemChannelFlags)), 'widget_channel_id': ('widget_channel', _transform_channel), 'rules_channel_id': ('rules_channel', _transform_channel), 'public_updates_channel_id': ('public_updates_channel', _transform_channel), 'permission_overwrites': ('overwrites', _transform_overwrites), 'splash_hash': ('splash', _guild_hash_transformer('splashes')), 'banner_hash': ('banner', _guild_hash_transformer('banners')), 'discovery_splash_hash': ('discovery_splash', _guild_hash_transformer('discovery-splashes')), 'icon_hash': ('icon', _transform_icon), 'avatar_hash': ('avatar', _transform_avatar), 'rate_limit_per_user': ('slowmode_delay', None), 'default_thread_rate_limit_per_user': ('default_thread_slowmode_delay', None), 'guild_id': ('guild', _transform_guild_id), 'tags': ('emoji', None), 'default_message_notifications': ('default_notifications', _enum_transformer(enums.NotificationLevel)), 'video_quality_mode': (None, _enum_transformer(enums.VideoQualityMode)), 'privacy_level': (None, _enum_transformer(enums.PrivacyLevel)), 'format_type': (None, _enum_transformer(enums.StickerFormatType)), 'type': (None, _transform_type), 'communication_disabled_until': ('timed_out_until', _transform_timestamp), 'expire_behavior': (None, _enum_transformer(enums.ExpireBehaviour)), 'mfa_level': (None, _enum_transformer(enums.MFALevel)), 'status': (None, _enum_transformer(enums.EventStatus)), 'entity_type': (None, _enum_transformer(enums.EntityType)), 'preferred_locale': (None, _enum_transformer(enums.Locale)), 'image_hash': ('cover_image', _transform_cover_image), 'trigger_type': (None, _enum_transformer(enums.AutoModRuleTriggerType)), 'event_type': (None, _enum_transformer(enums.AutoModRuleEventType)), 'actions': (None, _transform_automod_actions), 'exempt_channels': (None, _transform_channels_or_threads), 'exempt_roles': (None, _transform_roles), 'applied_tags': (None, _transform_applied_forum_tags), 'available_tags': (None, _transform_forum_tags), 'flags': (None, _transform_overloaded_flags), 'default_reaction_emoji': (None, _transform_default_reaction)}

    def __init__(self, entry: AuditLogEntry, data: List[AuditLogChangePayload]):
        if False:
            print('Hello World!')
        self.before: AuditLogDiff = AuditLogDiff()
        self.after: AuditLogDiff = AuditLogDiff()
        if entry.action is enums.AuditLogAction.app_command_permission_update:
            self.before.app_command_permissions = []
            self.after.app_command_permissions = []
            for elem in data:
                self._handle_app_command_permissions(self.before, entry, elem.get('old_value'))
                self._handle_app_command_permissions(self.after, entry, elem.get('new_value'))
            return
        for elem in data:
            attr = elem['key']
            if attr == '$add':
                self._handle_role(self.before, self.after, entry, elem['new_value'])
                continue
            elif attr == '$remove':
                self._handle_role(self.after, self.before, entry, elem['new_value'])
                continue
            if attr == 'trigger_metadata':
                self._handle_trigger_metadata(entry, elem, data)
                continue
            elif entry.action is enums.AuditLogAction.automod_rule_update and attr.startswith('$'):
                (action, _, trigger_attr) = attr.partition('_')
                if action == '$add':
                    self._handle_trigger_attr_update(self.before, self.after, entry, trigger_attr, elem['new_value'])
                elif action == '$remove':
                    self._handle_trigger_attr_update(self.after, self.before, entry, trigger_attr, elem['new_value'])
                continue
            try:
                (key, transformer) = self.TRANSFORMERS[attr]
            except (ValueError, KeyError):
                transformer = None
            else:
                if key:
                    attr = key
            transformer: Optional[Transformer]
            try:
                before = elem['old_value']
            except KeyError:
                before = None
            else:
                if transformer:
                    before = transformer(entry, before)
            setattr(self.before, attr, before)
            try:
                after = elem['new_value']
            except KeyError:
                after = None
            else:
                if transformer:
                    after = transformer(entry, after)
            setattr(self.after, attr, after)
        if hasattr(self.after, 'colour'):
            self.after.color = self.after.colour
            self.before.color = self.before.colour
        if hasattr(self.after, 'expire_behavior'):
            self.after.expire_behaviour = self.after.expire_behavior
            self.before.expire_behaviour = self.before.expire_behavior

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<AuditLogChanges before={self.before!r} after={self.after!r}>'

    def _handle_role(self, first: AuditLogDiff, second: AuditLogDiff, entry: AuditLogEntry, elem: List[RolePayload]) -> None:
        if False:
            print('Hello World!')
        if not hasattr(first, 'roles'):
            setattr(first, 'roles', [])
        data = []
        g: Guild = entry.guild
        for e in elem:
            role_id = int(e['id'])
            role = g.get_role(role_id)
            if role is None:
                role = Object(id=role_id, type=Role)
                role.name = e['name']
            data.append(role)
        setattr(second, 'roles', data)

    def _handle_app_command_permissions(self, diff: AuditLogDiff, entry: AuditLogEntry, data: Optional[ApplicationCommandPermissions]):
        if False:
            i = 10
            return i + 15
        if data is None:
            return
        from discord.app_commands import AppCommandPermissions
        state = entry._state
        guild = entry.guild
        diff.app_command_permissions.append(AppCommandPermissions(data=data, guild=guild, state=state))

    def _handle_trigger_metadata(self, entry: AuditLogEntry, data: AuditLogChangeTriggerMetadataPayload, full_data: List[AuditLogChangePayload]):
        if False:
            print('Hello World!')
        trigger_value: Optional[int] = None
        trigger_type: Optional[enums.AutoModRuleTriggerType] = None
        trigger_type = getattr(self.before, 'trigger_type', getattr(self.after, 'trigger_type', None))
        if trigger_type is None:
            if isinstance(entry.target, AutoModRule):
                trigger_value = entry.target.trigger.type.value
        else:
            trigger_value = trigger_type.value
        if trigger_value is None:
            _elem = utils.find(lambda elem: elem['key'] == 'trigger_type', full_data)
            if _elem is not None:
                trigger_value = _elem.get('old_value', _elem.get('new_value'))
            if trigger_value is None:
                combined = (data.get('old_value') or {}).keys() | (data.get('new_value') or {}).keys()
                if not combined:
                    trigger_value = enums.AutoModRuleTriggerType.spam.value
                elif 'presets' in combined:
                    trigger_value = enums.AutoModRuleTriggerType.keyword_preset.value
                elif 'keyword_filter' in combined or 'regex_patterns' in combined:
                    trigger_value = enums.AutoModRuleTriggerType.keyword.value
                elif 'mention_total_limit' in combined or 'mention_raid_protection_enabled' in combined:
                    trigger_value = enums.AutoModRuleTriggerType.mention_spam.value
                else:
                    trigger_value = -1
        self.before.trigger = AutoModTrigger.from_data(trigger_value, data.get('old_value'))
        self.after.trigger = AutoModTrigger.from_data(trigger_value, data.get('new_value'))

    def _handle_trigger_attr_update(self, first: AuditLogDiff, second: AuditLogDiff, entry: AuditLogEntry, attr: str, data: List[str]):
        if False:
            while True:
                i = 10
        self._create_trigger(first, entry)
        trigger = self._create_trigger(second, entry)
        try:
            getattr(trigger, attr).extend(data)
        except (AttributeError, TypeError):
            pass

    def _create_trigger(self, diff: AuditLogDiff, entry: AuditLogEntry) -> AutoModTrigger:
        if False:
            while True:
                i = 10
        if not hasattr(diff, 'trigger'):
            if isinstance(entry.target, AutoModRule):
                trigger_type = entry.target.trigger.type
            else:
                trigger_type = enums.try_enum(enums.AutoModRuleTriggerType, -1)
            diff.trigger = AutoModTrigger(type=trigger_type)
        return diff.trigger

class _AuditLogProxy:

    def __init__(self, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        for (k, v) in kwargs.items():
            setattr(self, k, v)

class _AuditLogProxyMemberPrune(_AuditLogProxy):
    delete_member_days: int
    members_removed: int

class _AuditLogProxyMemberMoveOrMessageDelete(_AuditLogProxy):
    channel: Union[abc.GuildChannel, Thread]
    count: int

class _AuditLogProxyMemberDisconnect(_AuditLogProxy):
    count: int

class _AuditLogProxyPinAction(_AuditLogProxy):
    channel: Union[abc.GuildChannel, Thread]
    message_id: int

class _AuditLogProxyStageInstanceAction(_AuditLogProxy):
    channel: abc.GuildChannel

class _AuditLogProxyMessageBulkDelete(_AuditLogProxy):
    count: int

class _AuditLogProxyAutoModAction(_AuditLogProxy):
    automod_rule_name: str
    automod_rule_trigger_type: str
    channel: Optional[Union[abc.GuildChannel, Thread]]

class _AuditLogProxyMemberKickOrMemberRoleUpdate(_AuditLogProxy):
    integration_type: Optional[str]

class AuditLogEntry(Hashable):
    """Represents an Audit Log entry.

    You retrieve these via :meth:`Guild.audit_logs`.

    .. container:: operations

        .. describe:: x == y

            Checks if two entries are equal.

        .. describe:: x != y

            Checks if two entries are not equal.

        .. describe:: hash(x)

            Returns the entry's hash.

    .. versionchanged:: 1.7
        Audit log entries are now comparable and hashable.

    Attributes
    -----------
    action: :class:`AuditLogAction`
        The action that was done.
    user: Optional[:class:`abc.User`]
        The user who initiated this action. Usually a :class:`Member`\\, unless gone
        then it's a :class:`User`.
    user_id: Optional[:class:`int`]
        The user ID who initiated this action.

        .. versionadded:: 2.2
    id: :class:`int`
        The entry ID.
    guild: :class:`Guild`
        The guild that this entry belongs to.
    target: Any
        The target that got changed. The exact type of this depends on
        the action being done.
    reason: Optional[:class:`str`]
        The reason this action was done.
    extra: Any
        Extra information that this entry has that might be useful.
        For most actions, this is ``None``. However in some cases it
        contains extra information. See :class:`AuditLogAction` for
        which actions have this field filled out.
    """

    def __init__(self, *, users: Mapping[int, User], integrations: Mapping[int, PartialIntegration], app_commands: Mapping[int, AppCommand], automod_rules: Mapping[int, AutoModRule], webhooks: Mapping[int, Webhook], data: AuditLogEntryPayload, guild: Guild):
        if False:
            for i in range(10):
                print('nop')
        self._state: ConnectionState = guild._state
        self.guild: Guild = guild
        self._users: Mapping[int, User] = users
        self._integrations: Mapping[int, PartialIntegration] = integrations
        self._app_commands: Mapping[int, AppCommand] = app_commands
        self._automod_rules: Mapping[int, AutoModRule] = automod_rules
        self._webhooks: Mapping[int, Webhook] = webhooks
        self._from_data(data)

    def _from_data(self, data: AuditLogEntryPayload) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.action: enums.AuditLogAction = enums.try_enum(enums.AuditLogAction, data['action_type'])
        self.id: int = int(data['id'])
        self.reason: Optional[str] = data.get('reason')
        extra = data.get('options')
        self.extra: Union[_AuditLogProxyMemberPrune, _AuditLogProxyMemberMoveOrMessageDelete, _AuditLogProxyMemberDisconnect, _AuditLogProxyPinAction, _AuditLogProxyStageInstanceAction, _AuditLogProxyMessageBulkDelete, _AuditLogProxyAutoModAction, _AuditLogProxyMemberKickOrMemberRoleUpdate, Member, User, None, PartialIntegration, Role, Object] = None
        if isinstance(self.action, enums.AuditLogAction) and extra:
            if self.action is enums.AuditLogAction.member_prune:
                self.extra = _AuditLogProxyMemberPrune(delete_member_days=int(extra['delete_member_days']), members_removed=int(extra['members_removed']))
            elif self.action is enums.AuditLogAction.member_move or self.action is enums.AuditLogAction.message_delete:
                channel_id = int(extra['channel_id'])
                self.extra = _AuditLogProxyMemberMoveOrMessageDelete(count=int(extra['count']), channel=self.guild.get_channel_or_thread(channel_id) or Object(id=channel_id))
            elif self.action is enums.AuditLogAction.member_disconnect:
                self.extra = _AuditLogProxyMemberDisconnect(count=int(extra['count']))
            elif self.action is enums.AuditLogAction.message_bulk_delete:
                self.extra = _AuditLogProxyMessageBulkDelete(count=int(extra['count']))
            elif self.action in (enums.AuditLogAction.kick, enums.AuditLogAction.member_role_update):
                integration_type = extra.get('integration_type')
                self.extra = _AuditLogProxyMemberKickOrMemberRoleUpdate(integration_type=integration_type)
            elif self.action.name.endswith('pin'):
                channel_id = int(extra['channel_id'])
                self.extra = _AuditLogProxyPinAction(channel=self.guild.get_channel_or_thread(channel_id) or Object(id=channel_id), message_id=int(extra['message_id']))
            elif self.action is enums.AuditLogAction.automod_block_message or self.action is enums.AuditLogAction.automod_flag_message or self.action is enums.AuditLogAction.automod_timeout_member:
                channel_id = utils._get_as_snowflake(extra, 'channel_id')
                channel = None
                if channel_id:
                    channel = self.guild.get_channel_or_thread(channel_id) or Object(id=channel_id)
                self.extra = _AuditLogProxyAutoModAction(automod_rule_name=extra['auto_moderation_rule_name'], automod_rule_trigger_type=enums.try_enum(enums.AutoModRuleTriggerType, extra['auto_moderation_rule_trigger_type']), channel=channel)
            elif self.action.name.startswith('overwrite_'):
                instance_id = int(extra['id'])
                the_type = extra.get('type')
                if the_type == '1':
                    self.extra = self._get_member(instance_id)
                elif the_type == '0':
                    role = self.guild.get_role(instance_id)
                    if role is None:
                        role = Object(id=instance_id, type=Role)
                        role.name = extra.get('role_name')
                    self.extra = role
            elif self.action.name.startswith('stage_instance'):
                channel_id = int(extra['channel_id'])
                self.extra = _AuditLogProxyStageInstanceAction(channel=self.guild.get_channel(channel_id) or Object(id=channel_id, type=StageChannel))
            elif self.action.name.startswith('app_command'):
                app_id = int(extra['application_id'])
                self.extra = self._get_integration_by_app_id(app_id) or Object(app_id, type=PartialIntegration)
        self._changes = data.get('changes', [])
        self.user_id: Optional[int] = utils._get_as_snowflake(data, 'user_id')
        self.user: Optional[Union[User, Member]] = self._get_member(self.user_id)
        self._target_id = utils._get_as_snowflake(data, 'target_id')

    def _get_member(self, user_id: Optional[int]) -> Union[Member, User, None]:
        if False:
            i = 10
            return i + 15
        if user_id is None:
            return None
        return self.guild.get_member(user_id) or self._users.get(user_id)

    def _get_integration(self, integration_id: Optional[int]) -> Optional[PartialIntegration]:
        if False:
            i = 10
            return i + 15
        if integration_id is None:
            return None
        return self._integrations.get(integration_id)

    def _get_integration_by_app_id(self, application_id: Optional[int]) -> Optional[PartialIntegration]:
        if False:
            while True:
                i = 10
        if application_id is None:
            return None
        return utils.get(self._integrations.values(), application_id=application_id)

    def _get_app_command(self, app_command_id: Optional[int]) -> Optional[AppCommand]:
        if False:
            for i in range(10):
                print('nop')
        if app_command_id is None:
            return None
        return self._app_commands.get(app_command_id)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<AuditLogEntry id={self.id} action={self.action} user={self.user!r}>'

    @utils.cached_property
    def created_at(self) -> datetime.datetime:
        if False:
            i = 10
            return i + 15
        ":class:`datetime.datetime`: Returns the entry's creation time in UTC."
        return utils.snowflake_time(self.id)

    @utils.cached_property
    def target(self) -> TargetType:
        if False:
            return 10
        if self.action.target_type is None:
            return None
        try:
            converter = getattr(self, '_convert_target_' + self.action.target_type)
        except AttributeError:
            if self._target_id is None:
                return None
            return Object(id=self._target_id)
        else:
            return converter(self._target_id)

    @utils.cached_property
    def category(self) -> Optional[enums.AuditLogActionCategory]:
        if False:
            return 10
        'Optional[:class:`AuditLogActionCategory`]: The category of the action, if applicable.'
        return self.action.category

    @utils.cached_property
    def changes(self) -> AuditLogChanges:
        if False:
            i = 10
            return i + 15
        ':class:`AuditLogChanges`: The list of changes this entry has.'
        obj = AuditLogChanges(self, self._changes)
        del self._changes
        return obj

    @utils.cached_property
    def before(self) -> AuditLogDiff:
        if False:
            for i in range(10):
                print('nop')
        ":class:`AuditLogDiff`: The target's prior state."
        return self.changes.before

    @utils.cached_property
    def after(self) -> AuditLogDiff:
        if False:
            return 10
        ":class:`AuditLogDiff`: The target's subsequent state."
        return self.changes.after

    def _convert_target_guild(self, target_id: int) -> Guild:
        if False:
            i = 10
            return i + 15
        return self.guild

    def _convert_target_channel(self, target_id: int) -> Union[abc.GuildChannel, Object]:
        if False:
            return 10
        return self.guild.get_channel(target_id) or Object(id=target_id)

    def _convert_target_user(self, target_id: Optional[int]) -> Optional[Union[Member, User, Object]]:
        if False:
            i = 10
            return i + 15
        if target_id is None:
            return None
        return self._get_member(target_id) or Object(id=target_id, type=Member)

    def _convert_target_role(self, target_id: int) -> Union[Role, Object]:
        if False:
            for i in range(10):
                print('nop')
        return self.guild.get_role(target_id) or Object(id=target_id, type=Role)

    def _convert_target_invite(self, target_id: None) -> Invite:
        if False:
            i = 10
            return i + 15
        changeset = self.before if self.action is enums.AuditLogAction.invite_delete else self.after
        fake_payload: InvitePayload = {'max_age': changeset.max_age, 'max_uses': changeset.max_uses, 'code': changeset.code, 'temporary': changeset.temporary, 'uses': changeset.uses, 'channel': None}
        obj = Invite(state=self._state, data=fake_payload, guild=self.guild, channel=changeset.channel)
        try:
            obj.inviter = changeset.inviter
        except AttributeError:
            pass
        return obj

    def _convert_target_emoji(self, target_id: int) -> Union[Emoji, Object]:
        if False:
            for i in range(10):
                print('nop')
        return self._state.get_emoji(target_id) or Object(id=target_id, type=Emoji)

    def _convert_target_message(self, target_id: int) -> Union[Member, User, Object]:
        if False:
            while True:
                i = 10
        return self._get_member(target_id) or Object(id=target_id, type=Member)

    def _convert_target_stage_instance(self, target_id: int) -> Union[StageInstance, Object]:
        if False:
            return 10
        return self.guild.get_stage_instance(target_id) or Object(id=target_id, type=StageInstance)

    def _convert_target_sticker(self, target_id: int) -> Union[GuildSticker, Object]:
        if False:
            for i in range(10):
                print('nop')
        return self._state.get_sticker(target_id) or Object(id=target_id, type=GuildSticker)

    def _convert_target_thread(self, target_id: int) -> Union[Thread, Object]:
        if False:
            for i in range(10):
                print('nop')
        return self.guild.get_thread(target_id) or Object(id=target_id, type=Thread)

    def _convert_target_guild_scheduled_event(self, target_id: int) -> Union[ScheduledEvent, Object]:
        if False:
            return 10
        return self.guild.get_scheduled_event(target_id) or Object(id=target_id, type=ScheduledEvent)

    def _convert_target_integration(self, target_id: int) -> Union[PartialIntegration, Object]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_integration(target_id) or Object(target_id, type=PartialIntegration)

    def _convert_target_app_command(self, target_id: int) -> Union[AppCommand, Object]:
        if False:
            for i in range(10):
                print('nop')
        target = self._get_app_command(target_id)
        if not target:
            from .app_commands import AppCommand
            target = Object(target_id, type=AppCommand)
        return target

    def _convert_target_integration_or_app_command(self, target_id: int) -> Union[PartialIntegration, AppCommand, Object]:
        if False:
            while True:
                i = 10
        target = self._get_integration_by_app_id(target_id) or self._get_app_command(target_id)
        if not target:
            try:
                from .app_commands import AppCommand
                target_app = self.extra
                app_id = target_app.application_id if isinstance(target_app, PartialIntegration) else target_app.id
                type = PartialIntegration if target_id == app_id else AppCommand
            except AttributeError:
                return Object(target_id)
            else:
                return Object(target_id, type=type)
        return target

    def _convert_target_auto_moderation(self, target_id: int) -> Union[AutoModRule, Object]:
        if False:
            for i in range(10):
                print('nop')
        return self._automod_rules.get(target_id) or Object(target_id, type=AutoModRule)

    def _convert_target_webhook(self, target_id: int) -> Union[Webhook, Object]:
        if False:
            return 10
        from .webhook import Webhook
        return self._webhooks.get(target_id) or Object(target_id, type=Webhook)