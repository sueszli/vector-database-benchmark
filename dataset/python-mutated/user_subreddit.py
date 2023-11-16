"""Provide the :class:`.UserSubreddit` class."""
from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, Any, Callable
from warnings import warn
from ...util.cache import cachedproperty
from .subreddit import Subreddit, SubredditModeration
if TYPE_CHECKING:
    import praw.models

class UserSubreddit(Subreddit):
    """A class for :class:`.User` Subreddits.

    To obtain an instance of this class execute:

    .. code-block:: python

        subreddit = reddit.user.me().subreddit

    .. include:: ../../typical_attributes.rst

    ========================= ==========================================================
    Attribute                 Description
    ========================= ==========================================================
    ``can_assign_link_flair`` Whether users can assign their own link flair.
    ``can_assign_user_flair`` Whether users can assign their own user flair.
    ``created_utc``           Time the subreddit was created, represented in `Unix
                              Time`_.
    ``description``           Subreddit description, in Markdown.
    ``description_html``      Subreddit description, in HTML.
    ``display_name``          Name of the subreddit.
    ``id``                    ID of the subreddit.
    ``name``                  Fullname of the subreddit.
    ``over18``                Whether the subreddit is NSFW.
    ``public_description``    Description of the subreddit, shown in searches and on the
                              "You must be invited to visit this community" page (if
                              applicable).
    ``spoilers_enabled``      Whether the spoiler tag feature is enabled.
    ``subscribers``           Count of subscribers. This will be ``0`` unless unless the
                              authenticated user is a moderator.
    ``user_is_banned``        Whether the authenticated user is banned.
    ``user_is_moderator``     Whether the authenticated user is a moderator.
    ``user_is_subscriber``    Whether the authenticated user is subscribed.
    ========================= ==========================================================

    .. _unix time: https://en.wikipedia.org/wiki/Unix_time

    """

    @staticmethod
    def _dict_deprecated_wrapper(func: Callable) -> Callable:
        if False:
            while True:
                i = 10
        'Show deprecation notice for dict only methods.'

        def wrapper(*args: Any, **kwargs: Any):
            if False:
                print('Hello World!')
            warn(f"'Redditor.subreddit' is no longer a dict and is now an UserSubreddit object. Using '{func.__name__}' is deprecated and will be removed in PRAW 8.", category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper

    @cachedproperty
    def mod(self) -> praw.models.reddit.user_subreddit.UserSubredditModeration:
        if False:
            i = 10
            return i + 15
        'Provide an instance of :class:`.UserSubredditModeration`.\n\n        For example, to update the authenticated user\'s display name:\n\n        .. code-block:: python\n\n            reddit.user.me().subreddit.mod.update(title="New display name")\n\n        '
        return UserSubredditModeration(self)

    def __getitem__(self, item: str) -> Any:
        if False:
            while True:
                i = 10
        'Show deprecation notice for dict method ``__getitem__``.'
        warn("'Redditor.subreddit' is no longer a dict and is now an UserSubreddit object. Accessing attributes using string indices is deprecated.", category=DeprecationWarning, stacklevel=2)
        return getattr(self, item)

    def __init__(self, reddit: praw.Reddit, *args: Any, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        'Initialize an :class:`.UserSubreddit` instance.\n\n        :param reddit: An instance of :class:`.Reddit`.\n\n        .. note::\n\n            This class should not be initialized directly. Instead, obtain an instance\n            via: ``reddit.user.me().subreddit`` or\n            ``reddit.redditor("redditor_name").subreddit``.\n\n        '

        def predicate(item: str):
            if False:
                i = 10
                return i + 15
            name = getattr(item, '__name__', None)
            return name not in dir(object) + dir(Subreddit) and name in dir(dict)
        for (name, _member) in inspect.getmembers(dict, predicate=predicate):
            if name != '__getitem__':
                setattr(self, name, self._dict_deprecated_wrapper(getattr(self.__dict__, name)))
        super().__init__(reddit, *args, **kwargs)

class UserSubredditModeration(SubredditModeration):
    """Provides a set of moderation functions to a :class:`.UserSubreddit`.

    For example, to accept a moderation invite from the user subreddit of u/spez:

    .. code-block:: python

        reddit.subreddit("test").mod.accept_invite()

    """

    def update(self, **settings: str | int | bool) -> dict[str, str | int | bool]:
        if False:
            return 10
        'Update the :class:`.Subreddit`\'s settings.\n\n        :param all_original_content: Mandate all submissions to be original content\n            only.\n        :param allow_chat_post_creation: Allow users to create chat submissions.\n        :param allow_images: Allow users to upload images using the native image\n            hosting.\n        :param allow_polls: Allow users to post polls to the subreddit.\n        :param allow_post_crossposts: Allow users to crosspost submissions from other\n            subreddits.\n        :param allow_top: Allow the subreddit to appear on r/all as well as the default\n            and trending lists.\n        :param allow_videos: Allow users to upload videos using the native image\n            hosting.\n        :param collapse_deleted_comments: Collapse deleted and removed comments on\n            comments pages by default.\n        :param crowd_control_chat_level: Controls the crowd control level for chat\n            rooms. Goes from 0-3.\n        :param crowd_control_level: Controls the crowd control level for submissions.\n            Goes from 0-3.\n        :param crowd_control_mode: Enables/disables crowd control.\n        :param comment_score_hide_mins: The number of minutes to hide comment scores.\n        :param description: Shown in the sidebar of your subreddit.\n        :param disable_contributor_requests: Specifies whether redditors may send\n            automated modmail messages requesting approval as a submitter.\n        :param exclude_banned_modqueue: Exclude posts by site-wide banned users from\n            modqueue/unmoderated.\n        :param free_form_reports: Allow users to specify custom reasons in the report\n            menu.\n        :param header_hover_text: The text seen when hovering over the snoo.\n        :param hide_ads: Don\'t show ads within this subreddit. Only applies to\n            Premium-user only subreddits.\n        :param key_color: A 6-digit rgb hex color (e.g., ``"#AABBCC"``), used as a\n            thematic color for your subreddit on mobile.\n        :param lang: A valid IETF language tag (underscore separated).\n        :param link_type: The types of submissions users can make. One of ``"any"``,\n            ``"link"``, or ``"self"``.\n        :param original_content_tag_enabled: Enables the use of the ``original content``\n            label for submissions.\n        :param over_18: Viewers must be over 18 years old (i.e., NSFW).\n        :param public_description: Public description blurb. Appears in search results\n            and on the landing page for private subreddits.\n        :param public_traffic: Make the traffic stats page public.\n        :param restrict_commenting: Specifies whether approved users have the ability to\n            comment.\n        :param restrict_posting: Specifies whether approved users have the ability to\n            submit posts.\n        :param show_media: Show thumbnails on submissions.\n        :param show_media_preview: Expand media previews on comments pages.\n        :param spam_comments: Spam filter strength for comments. One of ``"all"``,\n            ``"low"``, or ``"high"``.\n        :param spam_links: Spam filter strength for links. One of ``"all"``, ``"low"``,\n            or ``"high"``.\n        :param spam_selfposts: Spam filter strength for selfposts. One of ``"all"``,\n            ``"low"``, or ``"high"``.\n        :param spoilers_enabled: Enable marking posts as containing spoilers.\n        :param submit_link_label: Custom label for submit link button (None for\n            default).\n        :param submit_text: Text to show on submission page.\n        :param submit_text_label: Custom label for submit text post button (None for\n            default).\n        :param subreddit_type: The string ``"user"``.\n        :param suggested_comment_sort: All comment threads will use this sorting method\n            by default. Leave ``None``, or choose one of ``confidence``,\n            ``"controversial"``, ``"live"``, ``"new"``, ``"old"``, ``"qa"``,\n            ``"random"``, or ``"top"``.\n        :param title: The title of the subreddit.\n        :param welcome_message_enabled: Enables the subreddit welcome message.\n        :param welcome_message_text: The text to be used as a welcome message. A welcome\n            message is sent to all new subscribers by a Reddit bot.\n        :param wiki_edit_age: Account age, in days, required to edit and create wiki\n            pages.\n        :param wiki_edit_karma: Subreddit karma required to edit and create wiki pages.\n        :param wikimode: One of ``"anyone"``, ``"disabled"``, or ``"modonly"``.\n\n        Additional keyword arguments can be provided to handle new settings as Reddit\n        introduces them.\n\n        Settings that are documented here and aren\'t explicitly set by you in a call to\n        :meth:`.SubredditModeration.update` should retain their current value. If they\n        do not please file a bug.\n\n        .. warning::\n\n            Undocumented settings, or settings that were very recently documented, may\n            not retain their current value when updating. This often occurs when Reddit\n            adds a new setting but forgets to add that setting to the API endpoint that\n            is used to fetch the current settings.\n\n        '
        current_settings = self.settings()
        remap = {'allow_top': 'default_set', 'header_title': 'header_hover_text', 'lang': 'language', 'link_type': 'content_options', 'sr': 'subreddit_id', 'type': 'subreddit_type'}
        for (new, old) in remap.items():
            current_settings[new] = current_settings.pop(old)
        current_settings.update(settings)
        return UserSubreddit._create_or_update(_reddit=self.subreddit._reddit, **current_settings)