"""This module contains an object that represents a Telegram ChatPermission."""
from typing import TYPE_CHECKING, Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import Bot

class ChatPermissions(TelegramObject):
    """Describes actions that a non-administrator user is allowed to take in a chat.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`can_send_messages`,
    :attr:`can_send_polls`, :attr:`can_send_other_messages`, :attr:`can_add_web_page_previews`,
    :attr:`can_change_info`, :attr:`can_invite_users`, :attr:`can_pin_messages`,
    :attr:`can_send_audios`, :attr:`can_send_documents`, :attr:`can_send_photos`,
    :attr:`can_send_videos`, :attr:`can_send_video_notes`, :attr:`can_send_voice_notes`, and
    :attr:`can_manage_topics` are equal.

    .. versionchanged:: 20.0
        :attr:`can_manage_topics` is considered as well when comparing objects of
        this type in terms of equality.
    .. versionchanged:: 20.5

        * :attr:`can_send_audios`, :attr:`can_send_documents`, :attr:`can_send_photos`,
          :attr:`can_send_videos`, :attr:`can_send_video_notes` and :attr:`can_send_voice_notes`
          are considered as well when comparing objects of this type in terms of equality.
        * Removed deprecated argument and attribute ``can_send_media_messages``.


    Note:
        Though not stated explicitly in the official docs, Telegram changes not only the
        permissions that are set, but also sets all the others to :obj:`False`. However, since not
        documented, this behavior may change unbeknown to PTB.

    Args:
        can_send_messages (:obj:`bool`, optional): :obj:`True`, if the user is allowed to send text
            messages, contacts, locations and venues.
        can_send_polls (:obj:`bool`, optional): :obj:`True`, if the user is allowed to send polls.
        can_send_other_messages (:obj:`bool`, optional): :obj:`True`, if the user is allowed to
            send animations, games, stickers and use inline bots.
        can_add_web_page_previews (:obj:`bool`, optional): :obj:`True`, if the user is allowed to
            add web page previews to their messages.
        can_change_info (:obj:`bool`, optional): :obj:`True`, if the user is allowed to change the
            chat title, photo and other settings. Ignored in public supergroups.
        can_invite_users (:obj:`bool`, optional): :obj:`True`, if the user is allowed to invite new
            users to the chat.
        can_pin_messages (:obj:`bool`, optional): :obj:`True`, if the user is allowed to pin
            messages. Ignored in public supergroups.
        can_manage_topics (:obj:`bool`, optional): :obj:`True`, if the user is allowed
            to create forum topics. If omitted defaults to the value of
            :attr:`can_pin_messages`.

            .. versionadded:: 20.0
        can_send_audios (:obj:`bool`): :obj:`True`, if the user is allowed to send audios.

            .. versionadded:: 20.1
        can_send_documents (:obj:`bool`): :obj:`True`, if the user is allowed to send documents.

            .. versionadded:: 20.1
        can_send_photos (:obj:`bool`): :obj:`True`, if the user is allowed to send photos.

            .. versionadded:: 20.1
        can_send_videos (:obj:`bool`): :obj:`True`, if the user is allowed to send videos.

            .. versionadded:: 20.1
        can_send_video_notes (:obj:`bool`): :obj:`True`, if the user is allowed to send video
            notes.

            .. versionadded:: 20.1
        can_send_voice_notes (:obj:`bool`): :obj:`True`, if the user is allowed to send voice
            notes.

            .. versionadded:: 20.1

    Attributes:
        can_send_messages (:obj:`bool`): Optional. :obj:`True`, if the user is allowed to send text
            messages, contacts, locations and venues.
        can_send_polls (:obj:`bool`): Optional. :obj:`True`, if the user is allowed to send polls,
            implies :attr:`can_send_messages`.
        can_send_other_messages (:obj:`bool`): Optional. :obj:`True`, if the user is allowed to
            send animations, games, stickers and use inline bots.
        can_add_web_page_previews (:obj:`bool`): Optional. :obj:`True`, if the user is allowed to
            add web page previews to their messages.
        can_change_info (:obj:`bool`): Optional. :obj:`True`, if the user is allowed to change the
            chat title, photo and other settings. Ignored in public supergroups.
        can_invite_users (:obj:`bool`): Optional. :obj:`True`, if the user is allowed to invite
            new users to the chat.
        can_pin_messages (:obj:`bool`): Optional. :obj:`True`, if the user is allowed to pin
            messages. Ignored in public supergroups.
        can_manage_topics (:obj:`bool`): Optional. :obj:`True`, if the user is allowed
            to create forum topics. If omitted defaults to the value of
            :attr:`can_pin_messages`.

            .. versionadded:: 20.0
        can_send_audios (:obj:`bool`): :obj:`True`, if the user is allowed to send audios.

            .. versionadded:: 20.1
        can_send_documents (:obj:`bool`): :obj:`True`, if the user is allowed to send documents.

            .. versionadded:: 20.1
        can_send_photos (:obj:`bool`): :obj:`True`, if the user is allowed to send photos.

            .. versionadded:: 20.1
        can_send_videos (:obj:`bool`): :obj:`True`, if the user is allowed to send videos.

            .. versionadded:: 20.1
        can_send_video_notes (:obj:`bool`): :obj:`True`, if the user is allowed to send video
            notes.

            .. versionadded:: 20.1
        can_send_voice_notes (:obj:`bool`): :obj:`True`, if the user is allowed to send voice
            notes.

            .. versionadded:: 20.1

    """
    __slots__ = ('can_send_other_messages', 'can_invite_users', 'can_send_polls', 'can_send_messages', 'can_change_info', 'can_pin_messages', 'can_add_web_page_previews', 'can_manage_topics', 'can_send_audios', 'can_send_documents', 'can_send_photos', 'can_send_videos', 'can_send_video_notes', 'can_send_voice_notes')

    def __init__(self, can_send_messages: Optional[bool]=None, can_send_polls: Optional[bool]=None, can_send_other_messages: Optional[bool]=None, can_add_web_page_previews: Optional[bool]=None, can_change_info: Optional[bool]=None, can_invite_users: Optional[bool]=None, can_pin_messages: Optional[bool]=None, can_manage_topics: Optional[bool]=None, can_send_audios: Optional[bool]=None, can_send_documents: Optional[bool]=None, can_send_photos: Optional[bool]=None, can_send_videos: Optional[bool]=None, can_send_video_notes: Optional[bool]=None, can_send_voice_notes: Optional[bool]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            i = 10
            return i + 15
        super().__init__(api_kwargs=api_kwargs)
        self.can_send_messages: Optional[bool] = can_send_messages
        self.can_send_polls: Optional[bool] = can_send_polls
        self.can_send_other_messages: Optional[bool] = can_send_other_messages
        self.can_add_web_page_previews: Optional[bool] = can_add_web_page_previews
        self.can_change_info: Optional[bool] = can_change_info
        self.can_invite_users: Optional[bool] = can_invite_users
        self.can_pin_messages: Optional[bool] = can_pin_messages
        self.can_manage_topics: Optional[bool] = can_manage_topics
        self.can_send_audios: Optional[bool] = can_send_audios
        self.can_send_documents: Optional[bool] = can_send_documents
        self.can_send_photos: Optional[bool] = can_send_photos
        self.can_send_videos: Optional[bool] = can_send_videos
        self.can_send_video_notes: Optional[bool] = can_send_video_notes
        self.can_send_voice_notes: Optional[bool] = can_send_voice_notes
        self._id_attrs = (self.can_send_messages, self.can_send_polls, self.can_send_other_messages, self.can_add_web_page_previews, self.can_change_info, self.can_invite_users, self.can_pin_messages, self.can_manage_topics, self.can_send_audios, self.can_send_documents, self.can_send_photos, self.can_send_videos, self.can_send_video_notes, self.can_send_voice_notes)
        self._freeze()

    @classmethod
    def all_permissions(cls) -> 'ChatPermissions':
        if False:
            return 10
        '\n        This method returns an :class:`ChatPermissions` instance with all attributes\n        set to :obj:`True`. This is e.g. useful when unrestricting a chat member with\n        :meth:`telegram.Bot.restrict_chat_member`.\n\n        .. versionadded:: 20.0\n\n        '
        return cls(*14 * (True,))

    @classmethod
    def no_permissions(cls) -> 'ChatPermissions':
        if False:
            while True:
                i = 10
        '\n        This method returns an :class:`ChatPermissions` instance\n        with all attributes set to :obj:`False`.\n\n        .. versionadded:: 20.0\n        '
        return cls(*14 * (False,))

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['ChatPermissions']:
        if False:
            while True:
                i = 10
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        api_kwargs = {}
        if data.get('can_send_media_messages') is not None:
            api_kwargs['can_send_media_messages'] = data.pop('can_send_media_messages')
        return super()._de_json(data=data, bot=bot, api_kwargs=api_kwargs)