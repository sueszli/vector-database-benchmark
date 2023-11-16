"""Domain objects for user."""
from __future__ import annotations
import datetime
import re
from core import feconf
from core import utils
from core.constants import constants
from typing import Dict, List, Optional, TypedDict

class UserSettingsDict(TypedDict):
    """Dictionary representing the UserSettings object."""
    email: str
    roles: List[str]
    banned: bool
    has_viewed_lesson_info_modal_once: bool
    username: Optional[str]
    normalized_username: Optional[str]
    last_agreed_to_terms: Optional[datetime.datetime]
    last_started_state_editor_tutorial: Optional[datetime.datetime]
    last_started_state_translation_tutorial: Optional[datetime.datetime]
    last_logged_in: Optional[datetime.datetime]
    last_created_an_exploration: Optional[datetime.datetime]
    last_edited_an_exploration: Optional[datetime.datetime]
    default_dashboard: str
    creator_dashboard_display_pref: str
    user_bio: str
    subject_interests: List[str]
    first_contribution_msec: Optional[float]
    preferred_language_codes: List[str]
    preferred_site_language_code: Optional[str]
    preferred_audio_language_code: Optional[str]
    preferred_translation_language_code: Optional[str]
    pin: Optional[str]
    display_alias: Optional[str]
    deleted: bool
    created_on: Optional[datetime.datetime]

class UserSettings:
    """Value object representing a user's settings.

    Attributes:
        user_id: str. The unique ID of the user.
        email: str. The user email.
        roles: list(str). Roles of the user.
        has_viewed_lesson_info_modal_once: bool. Flag to check whether
            the user has viewed lesson info modal once which shows the progress
            of the user through exploration checkpoints.
        username: str or None. Identifiable username to display in the UI.
        last_agreed_to_terms: datetime.datetime or None. When the user last
            agreed to the terms of the site.
        last_started_state_editor_tutorial: datetime.datetime or None. When
            the user last started the state editor tutorial.
        last_started_state_translation_tutorial: datetime.datetime or None. When
            the user last started the state translation tutorial.
        last_logged_in: datetime.datetime or None. When the user last logged in.
        last_created_an_exploration: datetime.datetime or None. When the user
            last created an exploration.
        last_edited_an_exploration: datetime.datetime or None. When the user
            last edited an exploration.
        default_dashboard: str. The default dashboard of the user.
        user_bio: str. User-specified biography.
        subject_interests: list(str) or None. Subject interests specified by
            the user.
        first_contribution_msec: float or None. The time in milliseconds when
            the user first contributed to Oppia.
        preferred_language_codes: list(str) or None. Exploration language
            preferences specified by the user.
        preferred_site_language_code: str or None. System language preference.
        preferred_audio_language_code: str or None. Audio language preference.
        preferred_translation_language_code: str or None. Text Translation
            language preference of the translator that persists on the
            contributor dashboard.
        pin: str or None. The PIN of the user's profile for android.
        display_alias: str or None. Display name of a user who is logged
            into the Android app. None when the request is coming from web
            because we don't use it there.
    """

    def __init__(self, user_id: str, email: str, roles: List[str], banned: bool, has_viewed_lesson_info_modal_once: bool, username: Optional[str]=None, last_agreed_to_terms: Optional[datetime.datetime]=None, last_started_state_editor_tutorial: Optional[datetime.datetime]=None, last_started_state_translation_tutorial: Optional[datetime.datetime]=None, last_logged_in: Optional[datetime.datetime]=None, last_created_an_exploration: Optional[datetime.datetime]=None, last_edited_an_exploration: Optional[datetime.datetime]=None, default_dashboard: str=constants.DASHBOARD_TYPE_LEARNER, creator_dashboard_display_pref: str=constants.ALLOWED_CREATOR_DASHBOARD_DISPLAY_PREFS['CARD'], user_bio: str='', subject_interests: Optional[List[str]]=None, first_contribution_msec: Optional[float]=None, preferred_language_codes: Optional[List[str]]=None, preferred_site_language_code: Optional[str]=None, preferred_audio_language_code: Optional[str]=None, preferred_translation_language_code: Optional[str]=None, pin: Optional[str]=None, display_alias: Optional[str]=None, deleted: bool=False, created_on: Optional[datetime.datetime]=None) -> None:
        if False:
            while True:
                i = 10
        "Constructs a UserSettings domain object.\n\n        Args:\n            user_id: str. The unique ID of the user.\n            email: str. The user email.\n            roles: list(str). Roles of the user.\n            banned: bool. Whether the uses is banned.\n            has_viewed_lesson_info_modal_once: bool. Flag to check whether\n                the user has viewed lesson info modal once which shows the\n                progress of the user through exploration checkpoints.\n            username: str or None. Identifiable username to display in the UI.\n            last_agreed_to_terms: datetime.datetime or None. When the user\n                last agreed to the terms of the site.\n            last_started_state_editor_tutorial: datetime.datetime or None. When\n                the user last started the state editor tutorial.\n            last_started_state_translation_tutorial: datetime.datetime or None.\n                When the user last started the state translation tutorial.\n            last_logged_in: datetime.datetime or None. When the user last\n                logged in.\n            last_created_an_exploration: datetime.datetime or None. When the\n                user last created an exploration.\n            last_edited_an_exploration: datetime.datetime or None. When the\n                user last edited an exploration.\n            default_dashboard: str. The default dashboard of the user.\n            creator_dashboard_display_pref: str. The creator dashboard of the\n                user.\n            user_bio: str. User-specified biography.\n            subject_interests: list(str) or None. Subject interests specified by\n                the user.\n            first_contribution_msec: float or None. The time in milliseconds\n                when the user first contributed to Oppia.\n            preferred_language_codes: list(str) or None. Exploration language\n                preferences specified by the user.\n            preferred_site_language_code: str or None. System language\n                preference.\n            preferred_audio_language_code: str or None. Default language used\n                for audio translations preference.\n            preferred_translation_language_code: str or None. Text Translation\n                language preference of the translator that persists on the\n                contributor dashboard.\n            pin: str or None. The PIN of the user's profile for android.\n            display_alias: str or None. Display name of a user who is logged\n                into the Android app. None when the request is coming from\n                web because we don't use it there.\n            deleted: bool. Whether the user has requested removal of their\n                account.\n            created_on: datetime.datetime. When the user was created on.\n        "
        self.user_id = user_id
        self.email = email
        self.roles = roles
        self.username = username
        self.last_agreed_to_terms = last_agreed_to_terms
        self.last_started_state_editor_tutorial = last_started_state_editor_tutorial
        self.last_started_state_translation_tutorial = last_started_state_translation_tutorial
        self.last_logged_in = last_logged_in
        self.last_edited_an_exploration = last_edited_an_exploration
        self.last_created_an_exploration = last_created_an_exploration
        self.default_dashboard = default_dashboard
        self.creator_dashboard_display_pref = creator_dashboard_display_pref
        self.user_bio = user_bio
        self.subject_interests = subject_interests if subject_interests else []
        self.first_contribution_msec = first_contribution_msec
        self.preferred_language_codes = preferred_language_codes if preferred_language_codes else []
        self.preferred_site_language_code = preferred_site_language_code
        self.preferred_audio_language_code = preferred_audio_language_code
        self.preferred_translation_language_code = preferred_translation_language_code
        self.pin = pin
        self.display_alias = display_alias
        self.banned = banned
        self.deleted = deleted
        self.created_on = created_on
        self.has_viewed_lesson_info_modal_once = has_viewed_lesson_info_modal_once

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Checks that the user_id, email, roles, banned, pin and display_alias\n        fields of this UserSettings domain object are valid.\n\n        Raises:\n            ValidationError. The user_id is not str.\n            ValidationError. The email is not str.\n            ValidationError. The email is invalid.\n            ValidationError. The roles is not a list.\n            ValidationError. Given role does not exist.\n            ValidationError. The pin is not str.\n            ValidationError. The display alias is not str.\n        '
        if not isinstance(self.user_id, str):
            raise utils.ValidationError('Expected user_id to be a string, received %s' % self.user_id)
        if not self.user_id:
            raise utils.ValidationError('No user id specified.')
        if not utils.is_user_id_valid(self.user_id, allow_system_user_id=True, allow_pseudonymous_id=True):
            raise utils.ValidationError('The user ID is in a wrong format.')
        if not isinstance(self.banned, bool):
            raise utils.ValidationError('Expected banned to be a bool, received %s' % self.banned)
        if not isinstance(self.roles, list):
            raise utils.ValidationError('Expected roles to be a list, received %s' % self.roles)
        if self.banned:
            if self.roles:
                raise utils.ValidationError('Expected roles for banned user to be empty, recieved %s.' % self.roles)
        else:
            default_roles = []
            if len(self.roles) != len(set(self.roles)):
                raise utils.ValidationError('Roles contains duplicate values: %s' % self.roles)
            for role in self.roles:
                if not isinstance(role, str):
                    raise utils.ValidationError('Expected roles to be a string, received %s' % role)
                if role not in feconf.ALLOWED_USER_ROLES:
                    raise utils.ValidationError('Role %s does not exist.' % role)
                if role in feconf.ALLOWED_DEFAULT_USER_ROLES_ON_REGISTRATION:
                    default_roles.append(role)
            if len(default_roles) != 1:
                raise utils.ValidationError('Expected roles to contains one default role.')
        if self.pin is not None:
            if not isinstance(self.pin, str):
                raise utils.ValidationError('Expected PIN to be a string, received %s' % self.pin)
            if len(self.pin) != feconf.FULL_USER_PIN_LENGTH and len(self.pin) != feconf.PROFILE_USER_PIN_LENGTH:
                raise utils.ValidationError('User PIN can only be of length %s or %s' % (feconf.FULL_USER_PIN_LENGTH, feconf.PROFILE_USER_PIN_LENGTH))
            for character in self.pin:
                if character < '0' or character > '9':
                    raise utils.ValidationError('Only numeric characters are allowed in PIN.')
        if self.display_alias is not None and (not isinstance(self.display_alias, str)):
            raise utils.ValidationError('Expected display_alias to be a string, received %s' % self.display_alias)
        if not isinstance(self.email, str):
            raise utils.ValidationError('Expected email to be a string, received %s' % self.email)
        if not self.email:
            raise utils.ValidationError('No user email specified.')
        if '@' not in self.email or self.email.startswith('@') or self.email.endswith('@'):
            raise utils.ValidationError('Invalid email address: %s' % self.email)
        if not isinstance(self.creator_dashboard_display_pref, str):
            raise utils.ValidationError('Expected dashboard display preference to be a string, received %s' % self.creator_dashboard_display_pref)
        if self.creator_dashboard_display_pref not in list(constants.ALLOWED_CREATOR_DASHBOARD_DISPLAY_PREFS.values()):
            raise utils.ValidationError('%s is not a valid value for the dashboard display preferences.' % self.creator_dashboard_display_pref)

    def record_user_edited_an_exploration(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates last_edited_an_exploration to the current datetime for the\n        user.\n        '
        self.last_edited_an_exploration = datetime.datetime.utcnow()

    def update_first_contribution_msec(self, first_contribution_msec: float) -> None:
        if False:
            print('Hello World!')
        "Updates first_contribution_msec of user with given user_id\n        if it is set to None.\n\n        Args:\n            first_contribution_msec: float. New time to set in milliseconds\n                representing user's first contribution to Oppia.\n        "
        if self.first_contribution_msec is None:
            self.first_contribution_msec = first_contribution_msec

    def populate_from_modifiable_user_data(self, modifiable_user_data: ModifiableUserData) -> None:
        if False:
            print('Hello World!')
        'Populate the UserSettings domain object using the user data in\n            modifiable_user_data.\n\n        Args:\n            modifiable_user_data: ModifiableUserData. The modifiable user\n                data object with the information to be updated.\n\n        Raises:\n            ValidationError. None or empty value is provided for display alias\n                attribute.\n        '
        if not modifiable_user_data.display_alias or not isinstance(modifiable_user_data.display_alias, str):
            raise utils.ValidationError('Expected display_alias to be a string, received %s.' % modifiable_user_data.display_alias)
        self.display_alias = modifiable_user_data.display_alias
        self.preferred_language_codes = modifiable_user_data.preferred_language_codes
        self.preferred_site_language_code = modifiable_user_data.preferred_site_language_code
        self.preferred_audio_language_code = modifiable_user_data.preferred_audio_language_code
        self.preferred_translation_language_code = modifiable_user_data.preferred_translation_language_code
        self.pin = modifiable_user_data.pin

    def to_dict(self) -> UserSettingsDict:
        if False:
            while True:
                i = 10
        'Convert the UserSettings domain instance into a dictionary form\n        with its keys as the attributes of this class.\n\n        Returns:\n            dict. A dictionary containing the UserSettings class information\n            in a dictionary form.\n        '
        return {'email': self.email, 'roles': self.roles, 'banned': self.banned, 'username': self.username, 'normalized_username': self.normalized_username, 'last_agreed_to_terms': self.last_agreed_to_terms, 'last_started_state_editor_tutorial': self.last_started_state_editor_tutorial, 'last_started_state_translation_tutorial': self.last_started_state_translation_tutorial, 'last_logged_in': self.last_logged_in, 'last_edited_an_exploration': self.last_edited_an_exploration, 'last_created_an_exploration': self.last_created_an_exploration, 'default_dashboard': self.default_dashboard, 'creator_dashboard_display_pref': self.creator_dashboard_display_pref, 'user_bio': self.user_bio, 'subject_interests': self.subject_interests, 'first_contribution_msec': self.first_contribution_msec, 'preferred_language_codes': self.preferred_language_codes, 'preferred_site_language_code': self.preferred_site_language_code, 'preferred_audio_language_code': self.preferred_audio_language_code, 'preferred_translation_language_code': self.preferred_translation_language_code, 'pin': self.pin, 'display_alias': self.display_alias, 'deleted': self.deleted, 'created_on': self.created_on, 'has_viewed_lesson_info_modal_once': self.has_viewed_lesson_info_modal_once}

    @property
    def truncated_email(self) -> str:
        if False:
            print('Hello World!')
        'Returns truncated email by replacing last two characters before @\n        with period.\n\n        Returns:\n            str. The truncated email address of this UserSettings\n            domain object.\n        '
        first_part = self.email[:self.email.find('@')]
        last_part = self.email[self.email.find('@'):]
        if len(first_part) <= 1:
            first_part = '..'
        elif len(first_part) <= 3:
            first_part = '%s..' % first_part[0]
        else:
            first_part = first_part[:-3] + '..'
        return '%s%s' % (first_part, last_part)

    @property
    def normalized_username(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        "Returns username in lowercase or None if it does not exist.\n\n        Returns:\n            str or None. If this object has a 'username' property, returns\n            the normalized version of the username. Otherwise, returns None.\n        "
        if self.username:
            return self.normalize_username(self.username)
        else:
            return None

    @classmethod
    def normalize_username(cls, username: str) -> str:
        if False:
            while True:
                i = 10
        "Returns the normalized version of the given username,\n        or None if the passed-in 'username' is None.\n\n        Args:\n            username: str. Identifiable username to display in the UI.\n\n        Returns:\n            str. The normalized version of the given username.\n        "
        return username.lower()

    @classmethod
    def require_valid_username(cls, username: str) -> None:
        if False:
            print('Hello World!')
        'Checks if the given username is valid or not.\n\n        Args:\n            username: str. The username to validate.\n\n        Raises:\n            ValidationError. An empty username is supplied.\n            ValidationError. The given username exceeds the maximum allowed\n                number of characters.\n            ValidationError. The given username contains non-alphanumeric\n                characters.\n            ValidationError. The given username contains reserved substrings.\n        '
        if not username:
            raise utils.ValidationError('Empty username supplied.')
        if len(username) > constants.MAX_USERNAME_LENGTH:
            raise utils.ValidationError('A username can have at most %s characters.' % constants.MAX_USERNAME_LENGTH)
        if not re.match(feconf.ALPHANUMERIC_REGEX, username):
            raise utils.ValidationError('Usernames can only have alphanumeric characters.')
        reserved_usernames = set(feconf.SYSTEM_USERS.values()) | {'admin', 'oppia'}
        for reserved_username in reserved_usernames:
            if reserved_username in username.lower().strip():
                raise utils.ValidationError('This username is not available.')

    def mark_banned(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Marks a user banned.'
        self.banned = True
        self.roles = []

    def unmark_banned(self, default_role: str) -> None:
        if False:
            while True:
                i = 10
        'Unmarks ban for a banned user.\n\n        Args:\n            default_role: str. The role assigned to the user after marking\n                unbanned.\n        '
        self.banned = False
        self.roles = [default_role]

    def mark_lesson_info_modal_viewed(self) -> None:
        if False:
            i = 10
            return i + 15
        'Sets has_viewed_lesson_info_modal_once to true which shows\n        the user has viewed their progress through exploration in the lesson\n        info modal at least once in their lifetime journey.\n        '
        self.has_viewed_lesson_info_modal_once = True

class UserActionsInfo:
    """A class representing information of user actions.
    Attributes:
        user_id: str|None. The unique ID of the user, or None if the user
            is not logged in.
        roles: list(str). The roles of the user.
        actions: list(str). A list of actions accessible to the role.
    """

    def __init__(self, user_id: Optional[str], roles: List[str], actions: List[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._user_id = user_id
        self._roles = roles
        self._actions = actions

    @property
    def user_id(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the unique ID of the user.\n\n        Returns:\n            user_id: str. The unique ID of the user.\n        '
        return self._user_id

    @property
    def roles(self) -> List[str]:
        if False:
            while True:
                i = 10
        'Returns the roles of user.\n\n        Returns:\n            role: list(str). The roles of the user.\n        '
        return self._roles

    @property
    def actions(self) -> List[str]:
        if False:
            while True:
                i = 10
        'Returns list of actions accessible to a user.\n\n        Returns:\n            actions: list(str). List of actions accessible to a user ID.\n        '
        return self._actions

class UserContributions:
    """Value object representing a user's contributions.

    Attributes:
        user_id: str. The unique ID of the user.
        created_exploration_ids: list(str). IDs of explorations that this
            user has created.
        edited_exploration_ids: list(str). IDs of explorations that this
            user has edited.
    """

    def __init__(self, user_id: str, created_exploration_ids: List[str], edited_exploration_ids: List[str]) -> None:
        if False:
            return 10
        'Constructs a UserContributions domain object.\n\n        Args:\n            user_id: str. The unique ID of the user.\n            created_exploration_ids: list(str). IDs of explorations that this\n                user has created.\n            edited_exploration_ids: list(str). IDs of explorations that this\n                user has edited.\n        '
        self.user_id = user_id
        self.created_exploration_ids = created_exploration_ids
        self.edited_exploration_ids = edited_exploration_ids

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Checks that user_id, created_exploration_ids and\n        edited_exploration_ids fields of this UserContributions\n        domain object are valid.\n\n        Raises:\n            ValidationError. No user id specified.\n            ValidationError. The user_id is not str.\n            ValidationError. The created_exploration_ids is not a list.\n            ValidationError. The exploration_id in created_exploration_ids\n                is not str.\n            ValidationError. The edited_exploration_ids is not a list.\n            ValidationError. The exploration_id in edited_exploration_ids\n                is not str.\n        '
        if not isinstance(self.user_id, str):
            raise utils.ValidationError('Expected user_id to be a string, received %s' % self.user_id)
        if not self.user_id:
            raise utils.ValidationError('No user id specified.')
        if not isinstance(self.created_exploration_ids, list):
            raise utils.ValidationError('Expected created_exploration_ids to be a list, received %s' % self.created_exploration_ids)
        for exploration_id in self.created_exploration_ids:
            if not isinstance(exploration_id, str):
                raise utils.ValidationError('Expected exploration_id in created_exploration_ids to be a string, received %s' % exploration_id)
        if not isinstance(self.edited_exploration_ids, list):
            raise utils.ValidationError('Expected edited_exploration_ids to be a list, received %s' % self.edited_exploration_ids)
        for exploration_id in self.edited_exploration_ids:
            if not isinstance(exploration_id, str):
                raise utils.ValidationError('Expected exploration_id in edited_exploration_ids to be a string, received %s' % exploration_id)

    def add_created_exploration_id(self, exploration_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Adds an exploration_id to list of created explorations.\n\n        Args:\n            exploration_id: str. The exploration id.\n        '
        if exploration_id not in self.created_exploration_ids:
            self.created_exploration_ids.append(exploration_id)
            self.created_exploration_ids.sort()

    def add_edited_exploration_id(self, exploration_id: str) -> None:
        if False:
            print('Hello World!')
        'Adds an exploration_id to list of edited explorations.\n\n        Args:\n            exploration_id: str. The exploration id.\n        '
        if exploration_id not in self.edited_exploration_ids:
            self.edited_exploration_ids.append(exploration_id)
            self.edited_exploration_ids.sort()

class UserGlobalPrefs:
    """Domain object for user global email preferences.

    Attributes:
        can_receive_email_updates: bool. Whether the user can receive
            email updates.
        can_receive_editor_role_email: bool. Whether the user can receive
            emails notifying them of role changes.
        can_receive_feedback_message_email: bool. Whether the user can
            receive emails when users submit feedback to their explorations.
        can_receive_subscription_email: bool. Whether the user can receive
             subscription emails notifying them about new explorations.
    """

    def __init__(self, can_receive_email_updates: bool, can_receive_editor_role_email: bool, can_receive_feedback_message_email: bool, can_receive_subscription_email: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs a UserGlobalPrefs domain object.\n\n        Args:\n            can_receive_email_updates: bool. Whether the user can receive\n                email updates.\n            can_receive_editor_role_email: bool. Whether the user can receive\n                emails notifying them of role changes.\n            can_receive_feedback_message_email: bool. Whether the user can\n                receive emails when users submit feedback to their explorations.\n            can_receive_subscription_email: bool. Whether the user can receive\n                subscription emails notifying them about new explorations.\n        '
        self.can_receive_email_updates = can_receive_email_updates
        self.can_receive_editor_role_email = can_receive_editor_role_email
        self.can_receive_feedback_message_email = can_receive_feedback_message_email
        self.can_receive_subscription_email = can_receive_subscription_email

    @classmethod
    def create_default_prefs(cls) -> UserGlobalPrefs:
        if False:
            i = 10
            return i + 15
        'Returns UserGlobalPrefs with default attributes.'
        return cls(feconf.DEFAULT_EMAIL_UPDATES_PREFERENCE, feconf.DEFAULT_EDITOR_ROLE_EMAIL_PREFERENCE, feconf.DEFAULT_FEEDBACK_MESSAGE_EMAIL_PREFERENCE, feconf.DEFAULT_SUBSCRIPTION_EMAIL_PREFERENCE)

class UserExplorationPrefsDict(TypedDict):
    """Dictionary representing the UserExplorationPrefs object."""
    mute_feedback_notifications: bool
    mute_suggestion_notifications: bool

class UserExplorationPrefs:
    """Domain object for user exploration email preferences.

    Attributes:
        mute_feedback_notifications: bool. Whether the given user has muted
            feedback emails.
        mute_suggestion_notifications: bool. Whether the given user has
            muted suggestion emails.
    """

    def __init__(self, mute_feedback_notifications: bool, mute_suggestion_notifications: bool) -> None:
        if False:
            i = 10
            return i + 15
        'Constructs a UserExplorationPrefs domain object.\n\n        Args:\n            mute_feedback_notifications: bool. Whether the given user has muted\n                feedback emails.\n            mute_suggestion_notifications: bool. Whether the given user has\n                muted suggestion emails.\n        '
        self.mute_feedback_notifications = mute_feedback_notifications
        self.mute_suggestion_notifications = mute_suggestion_notifications

    @classmethod
    def create_default_prefs(cls) -> UserExplorationPrefs:
        if False:
            return 10
        'Returns UserExplorationPrefs with default attributes.'
        return cls(feconf.DEFAULT_FEEDBACK_NOTIFICATIONS_MUTED_PREFERENCE, feconf.DEFAULT_SUGGESTION_NOTIFICATIONS_MUTED_PREFERENCE)

    def to_dict(self) -> UserExplorationPrefsDict:
        if False:
            while True:
                i = 10
        "Return dictionary representation of UserExplorationPrefs.\n\n        Returns:\n            dict. The keys of the dict are:\n                'mute_feedback_notifications': bool. Whether the given user has\n                    muted feedback emails.\n                'mute_suggestion_notifications': bool. Whether the given user\n                    has muted suggestion emails.\n        "
        return {'mute_feedback_notifications': self.mute_feedback_notifications, 'mute_suggestion_notifications': self.mute_suggestion_notifications}

class ExpUserLastPlaythrough:
    """Domain object for an exploration last playthrough model."""

    def __init__(self, user_id: str, exploration_id: str, last_played_exp_version: int, last_updated: datetime.datetime, last_played_state_name: str) -> None:
        if False:
            while True:
                i = 10
        self.id = '%s.%s' % (user_id, exploration_id)
        self.user_id = user_id
        self.exploration_id = exploration_id
        self.last_played_exp_version = last_played_exp_version
        self.last_updated = last_updated
        self.last_played_state_name = last_played_state_name

    def update_last_played_information(self, last_played_exp_version: int, last_played_state_name: str) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the last playthrough information of the user.\n\n        Args:\n            last_played_exp_version: int. The version of the exploration that\n                was played by the user.\n            last_played_state_name: str. The name of the state at which the\n                learner left the exploration.\n        '
        self.last_played_exp_version = last_played_exp_version
        self.last_played_state_name = last_played_state_name

class IncompleteActivities:
    """Domain object for the incomplete activities model."""

    def __init__(self, user_id: str, exploration_ids: List[str], collection_ids: List[str], story_ids: List[str], partially_learnt_topic_ids: List[str], partially_mastered_topic_id: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.id = user_id
        self.exploration_ids = exploration_ids
        self.collection_ids = collection_ids
        self.story_ids = story_ids
        self.partially_learnt_topic_ids = partially_learnt_topic_ids
        self.partially_mastered_topic_id = partially_mastered_topic_id

    def add_exploration_id(self, exploration_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adds the exploration id to the list of incomplete exploration ids.\n\n        Args:\n            exploration_id: str. The exploration id to be inserted into the\n                incomplete list.\n        '
        self.exploration_ids.append(exploration_id)

    def remove_exploration_id(self, exploration_id: str) -> None:
        if False:
            while True:
                i = 10
        'Removes the exploration id from the list of incomplete exploration\n        ids.\n\n        Args:\n            exploration_id: str. The exploration id to be removed from the\n                incomplete list.\n        '
        self.exploration_ids.remove(exploration_id)

    def add_collection_id(self, collection_id: str) -> None:
        if False:
            return 10
        'Adds the collection id to the list of incomplete collection ids.\n\n        Args:\n            collection_id: str. The collection id to be inserted into the\n                incomplete list.\n        '
        self.collection_ids.append(collection_id)

    def remove_collection_id(self, collection_id: str) -> None:
        if False:
            return 10
        'Removes the collection id from the list of incomplete collection\n        ids.\n\n        Args:\n            collection_id: str. The collection id to be removed from the\n                incomplete list.\n        '
        self.collection_ids.remove(collection_id)

    def add_story_id(self, story_id: str) -> None:
        if False:
            return 10
        'Adds the story id to the list of incomplete story ids.\n\n        Args:\n            story_id: str. The story id to be inserted into the\n                incomplete list.\n        '
        self.story_ids.append(story_id)

    def remove_story_id(self, story_id: str) -> None:
        if False:
            print('Hello World!')
        'Removes the story id from the list of incomplete story\n        ids.\n\n        Args:\n            story_id: str. The story id to be removed from the\n                incomplete list.\n        '
        self.story_ids.remove(story_id)

    def add_partially_learnt_topic_id(self, partially_learnt_topic_id: str) -> None:
        if False:
            while True:
                i = 10
        'Adds the topic id to the list of partially learnt topic ids.\n\n        Args:\n            partially_learnt_topic_id: str. The topic id to be inserted in the\n                partially learnt list.\n        '
        self.partially_learnt_topic_ids.append(partially_learnt_topic_id)

    def remove_partially_learnt_topic_id(self, partially_learnt_topic_id: str) -> None:
        if False:
            while True:
                i = 10
        'Removes the topic id from the list of partially learnt topic\n        ids.\n\n        Args:\n            partially_learnt_topic_id: str. The topic id to be removed from the\n                partially learnt list.\n        '
        self.partially_learnt_topic_ids.remove(partially_learnt_topic_id)

class CompletedActivities:
    """Domain object for the activities completed by learner model."""

    def __init__(self, user_id: str, exploration_ids: List[str], collection_ids: List[str], story_ids: List[str], learnt_topic_ids: List[str], mastered_topic_ids: Optional[List[str]]=None) -> None:
        if False:
            return 10
        self.id = user_id
        self.exploration_ids = exploration_ids
        self.collection_ids = collection_ids
        self.story_ids = story_ids
        self.learnt_topic_ids = learnt_topic_ids
        self.mastered_topic_ids = mastered_topic_ids

    def add_exploration_id(self, exploration_id: str) -> None:
        if False:
            print('Hello World!')
        'Adds the exploration id to the list of completed exploration ids.\n\n        Args:\n            exploration_id: str. The exploration id to be inserted into the\n                completed list.\n        '
        self.exploration_ids.append(exploration_id)

    def remove_exploration_id(self, exploration_id: str) -> None:
        if False:
            return 10
        'Removes the exploration id from the list of completed exploration\n        ids.\n\n        Args:\n            exploration_id: str. The exploration id to be removed from the\n                completed list.\n        '
        self.exploration_ids.remove(exploration_id)

    def add_collection_id(self, collection_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Adds the collection id to the list of completed collection ids.\n\n        Args:\n            collection_id: str. The collection id to be inserted into the\n                completed list.\n        '
        self.collection_ids.append(collection_id)

    def remove_collection_id(self, collection_id: str) -> None:
        if False:
            return 10
        'Removes the collection id from the list of completed collection\n        ids.\n\n        Args:\n            collection_id: str. The collection id to be removed from the\n                completed list.\n        '
        self.collection_ids.remove(collection_id)

    def add_story_id(self, story_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Adds the story id to the list of completed story ids.\n\n        Args:\n            story_id: str. The story id to be inserted in the\n                completed list.\n        '
        self.story_ids.append(story_id)

    def remove_story_id(self, story_id: str) -> None:
        if False:
            while True:
                i = 10
        'Removes the story id from the list of completed story\n        ids.\n\n        Args:\n            story_id: str. The story id to be removed from the\n                completed list.\n        '
        self.story_ids.remove(story_id)

    def add_learnt_topic_id(self, learnt_topic_id: str) -> None:
        if False:
            print('Hello World!')
        'Adds the topic id to the list of learnt topic ids.\n\n        Args:\n            learnt_topic_id: str. The topic id to be inserted in the\n                learnt list.\n        '
        self.learnt_topic_ids.append(learnt_topic_id)

    def remove_learnt_topic_id(self, learnt_topic_id: str) -> None:
        if False:
            while True:
                i = 10
        'Removes the topic id from the list of learnt topic\n        ids.\n\n        Args:\n            learnt_topic_id: str. The topic id to be removed from the\n                learnt list.\n        '
        self.learnt_topic_ids.remove(learnt_topic_id)

class LearnerGoalsDict(TypedDict):
    """Dictionary representing the LearnerGoals object."""
    topic_ids_to_learn: List[str]
    topic_ids_to_master: List[str]

class LearnerGoals:
    """Domain object for the learner goals model."""

    def __init__(self, user_id: str, topic_ids_to_learn: List[str], topic_ids_to_master: List[str]) -> None:
        if False:
            while True:
                i = 10
        self.id = user_id
        self.topic_ids_to_learn = topic_ids_to_learn
        self.topic_ids_to_master = topic_ids_to_master

    def add_topic_id_to_learn(self, topic_id: str) -> None:
        if False:
            i = 10
            return i + 15
        "Adds the topic id to 'topic IDs to learn' list.\n\n        Args:\n            topic_id: str. The topic id to be inserted to the learn list.\n        "
        self.topic_ids_to_learn.append(topic_id)

    def remove_topic_id_from_learn(self, topic_id: str) -> None:
        if False:
            i = 10
            return i + 15
        "Removes the topic id from the 'topic IDs to learn' list.\n\n        topic_id: str. The id of the topic to be removed.\n        "
        self.topic_ids_to_learn.remove(topic_id)

    def to_dict(self) -> LearnerGoalsDict:
        if False:
            for i in range(10):
                print('nop')
        'Return dictionary representation of LearnerGoals.\n\n        Returns:\n            dict. A dictionary containing the LearnerGoals class information\n            in a dictionary form.\n        '
        return {'topic_ids_to_learn': self.topic_ids_to_learn, 'topic_ids_to_master': self.topic_ids_to_master}

class LearnerPlaylist:
    """Domain object for the learner playlist model."""

    def __init__(self, user_id: str, exploration_ids: List[str], collection_ids: List[str]) -> None:
        if False:
            i = 10
            return i + 15
        self.id = user_id
        self.exploration_ids = exploration_ids
        self.collection_ids = collection_ids

    def insert_exploration_id_at_given_position(self, exploration_id: str, position_to_be_inserted: int) -> None:
        if False:
            print('Hello World!')
        'Inserts the given exploration id at the given position.\n\n        Args:\n            exploration_id: str. The exploration id to be inserted into the\n                play later list.\n            position_to_be_inserted: int. The position at which it\n                is to be inserted.\n        '
        self.exploration_ids.insert(position_to_be_inserted, exploration_id)

    def add_exploration_id_to_list(self, exploration_id: str) -> None:
        if False:
            print('Hello World!')
        'Inserts the exploration id at the end of the list.\n\n        Args:\n            exploration_id: str. The exploration id to be appended to the end\n                of the list.\n        '
        self.exploration_ids.append(exploration_id)

    def insert_collection_id_at_given_position(self, collection_id: str, position_to_be_inserted: int) -> None:
        if False:
            while True:
                i = 10
        'Inserts the given collection id at the given position.\n\n        Args:\n            collection_id: str. The collection id to be inserted into the\n                play later list.\n            position_to_be_inserted: int. The position at which it\n                is to be inserted.\n        '
        self.collection_ids.insert(position_to_be_inserted, collection_id)

    def add_collection_id_to_list(self, collection_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Inserts the collection id at the end of the list.\n\n        Args:\n            collection_id: str. The collection id to be appended to the end\n                of the list.\n        '
        self.collection_ids.append(collection_id)

    def remove_exploration_id(self, exploration_id: str) -> None:
        if False:
            while True:
                i = 10
        'Removes the exploration id from the learner playlist.\n\n        exploration_id: str. The id of the exploration to be removed.\n        '
        self.exploration_ids.remove(exploration_id)

    def remove_collection_id(self, collection_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Removes the collection id from the learner playlist.\n\n        collection_id: str. The id of the collection to be removed.\n        '
        self.collection_ids.remove(collection_id)

class UserContributionProficiency:
    """Domain object for UserContributionProficiencyModel."""

    def __init__(self, user_id: str, score_category: str, score: int, onboarding_email_sent: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user_id = user_id
        self.score_category = score_category
        self.score = score
        self.onboarding_email_sent = onboarding_email_sent

    def increment_score(self, increment_by: int) -> None:
        if False:
            print('Hello World!')
        "Increments the score of the user in the category by the given amount.\n\n        In the first version of the scoring system, the increment_by quantity\n        will be +1, i.e, each user gains a point for a successful contribution\n        and doesn't lose score in any way.\n\n        Args:\n            increment_by: float. The amount to increase the score of the user\n                by.\n        "
        self.score += increment_by

    def can_user_review_category(self) -> bool:
        if False:
            print('Hello World!')
        'Checks if user can review suggestions in category score_category.\n        If the user has score above the minimum required score, then the user\n        is allowed to review.\n\n        Returns:\n            bool. Whether the user can review suggestions under category\n            score_category.\n        '
        return self.score >= feconf.MINIMUM_SCORE_REQUIRED_TO_REVIEW

    def mark_onboarding_email_as_sent(self) -> None:
        if False:
            return 10
        'Marks the email as sent.'
        self.onboarding_email_sent = True

class UserContributionRights:
    """Domain object for the UserContributionRightsModel."""

    def __init__(self, user_id: str, can_review_translation_for_language_codes: List[str], can_review_voiceover_for_language_codes: List[str], can_review_questions: bool, can_submit_questions: bool):
        if False:
            print('Hello World!')
        self.id = user_id
        self.can_review_translation_for_language_codes = can_review_translation_for_language_codes
        self.can_review_voiceover_for_language_codes = can_review_voiceover_for_language_codes
        self.can_review_questions = can_review_questions
        self.can_submit_questions = can_submit_questions

    def can_review_at_least_one_item(self) -> bool:
        if False:
            print('Hello World!')
        'Checks whether user has rights to review at least one item.\n\n        Returns:\n            boolean. Whether user has rights to review at east one item.\n        '
        return bool(self.can_review_translation_for_language_codes or self.can_review_voiceover_for_language_codes or self.can_review_questions)

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Validates different attributes of the class.'
        if not isinstance(self.can_review_translation_for_language_codes, list):
            raise utils.ValidationError('Expected can_review_translation_for_language_codes to be a list, found: %s' % type(self.can_review_translation_for_language_codes))
        for language_code in self.can_review_translation_for_language_codes:
            if not utils.is_supported_audio_language_code(language_code):
                raise utils.ValidationError('Invalid language_code: %s' % language_code)
        if len(self.can_review_translation_for_language_codes) != len(set(self.can_review_translation_for_language_codes)):
            raise utils.ValidationError('Expected can_review_translation_for_language_codes list not to have duplicate values, found: %s' % self.can_review_translation_for_language_codes)
        if not isinstance(self.can_review_voiceover_for_language_codes, list):
            raise utils.ValidationError('Expected can_review_voiceover_for_language_codes to be a list, found: %s' % type(self.can_review_voiceover_for_language_codes))
        for language_code in self.can_review_voiceover_for_language_codes:
            if not utils.is_supported_audio_language_code(language_code):
                raise utils.ValidationError('Invalid language_code: %s' % language_code)
        if len(self.can_review_voiceover_for_language_codes) != len(set(self.can_review_voiceover_for_language_codes)):
            raise utils.ValidationError('Expected can_review_voiceover_for_language_codes list not to have duplicate values, found: %s' % self.can_review_voiceover_for_language_codes)
        if not isinstance(self.can_review_questions, bool):
            raise utils.ValidationError('Expected can_review_questions to be a boolean value, found: %s' % type(self.can_review_questions))
        if not isinstance(self.can_submit_questions, bool):
            raise utils.ValidationError('Expected can_submit_questions to be a boolean value, found: %s' % type(self.can_submit_questions))

class ModifiableUserDataDict(TypedDict):
    """Dictionary representing the ModifiableUserData object."""
    display_alias: str
    pin: Optional[str]
    preferred_language_codes: List[str]
    preferred_site_language_code: Optional[str]
    preferred_audio_language_code: Optional[str]
    preferred_translation_language_code: Optional[str]
    user_id: Optional[str]

class RawUserDataDict(TypedDict):
    """Type for the argument raw_user_data_dict."""
    schema_version: int
    display_alias: str
    pin: Optional[str]
    preferred_language_codes: List[str]
    preferred_site_language_code: Optional[str]
    preferred_audio_language_code: Optional[str]
    preferred_translation_language_code: Optional[str]
    user_id: Optional[str]

class ModifiableUserData:
    """Domain object to represent the new values in a UserSettingsModel change
    submitted by the Android client.
    """

    def __init__(self, display_alias: str, pin: Optional[str], preferred_language_codes: List[str], preferred_site_language_code: Optional[str], preferred_audio_language_code: Optional[str], preferred_translation_language_code: Optional[str], user_id: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        "Constructs a ModifiableUserData domain object.\n\n        Args:\n            display_alias: str. Display alias of the user shown on Android.\n            pin: str or None. PIN of the user used for PIN based authentication\n                on Android. None if it hasn't been set till now.\n            preferred_language_codes: list(str). Exploration language\n                preferences specified by the user.\n            preferred_site_language_code: str or None. System language\n                preference.\n            preferred_audio_language_code: str or None. Audio language\n                preference.\n            preferred_translation_language_code: str or None. Text Translation\n                language preference of the translator that persists on the\n                contributor dashboard.\n            user_id: str or None. User ID of the user whose data is being\n                updated. None if request did not have a user_id for the user\n                yet and expects the backend to create a new user entry for it.\n        "
        self.display_alias = display_alias
        self.pin = pin
        self.preferred_language_codes = preferred_language_codes
        self.preferred_site_language_code = preferred_site_language_code
        self.preferred_audio_language_code = preferred_audio_language_code
        self.preferred_translation_language_code = preferred_translation_language_code
        self.user_id = user_id

    @classmethod
    def from_dict(cls, modifiable_user_data_dict: ModifiableUserDataDict) -> ModifiableUserData:
        if False:
            for i in range(10):
                print('nop')
        'Return a ModifiableUserData domain object from a dict.\n\n        Args:\n            modifiable_user_data_dict: dict. The dict representation of\n                ModifiableUserData object.\n\n        Returns:\n            ModifiableUserData. The corresponding ModifiableUserData domain\n            object.\n        '
        return ModifiableUserData(modifiable_user_data_dict['display_alias'], modifiable_user_data_dict['pin'], modifiable_user_data_dict['preferred_language_codes'], modifiable_user_data_dict['preferred_site_language_code'], modifiable_user_data_dict['preferred_audio_language_code'], modifiable_user_data_dict['preferred_translation_language_code'], modifiable_user_data_dict['user_id'])
    CURRENT_SCHEMA_VERSION = 1

    @classmethod
    def from_raw_dict(cls, raw_user_data_dict: RawUserDataDict) -> ModifiableUserData:
        if False:
            print('Hello World!')
        'Converts the raw_user_data_dict into a ModifiableUserData domain\n        object by converting it according to the latest schema format.\n\n        Args:\n            raw_user_data_dict: dict. The input raw form of user_data dict\n                coming from the controller layer, which has to be converted.\n\n        Returns:\n            ModifiableUserData. The domain object representing the user data\n            dict transformed according to the latest schema version.\n\n        Raises:\n            Exception. No schema version specified.\n            Exception. Schema version is not of type int.\n            Exception. Invalid schema version.\n        '
        data_schema_version = raw_user_data_dict['schema_version']
        if data_schema_version is None:
            raise Exception('Invalid modifiable user data: no schema version specified.')
        if not isinstance(data_schema_version, int):
            raise Exception('Version has invalid type, expected int, received %s' % type(data_schema_version))
        if not isinstance(data_schema_version, int) or data_schema_version < 1 or data_schema_version > cls.CURRENT_SCHEMA_VERSION:
            raise Exception('Invalid version %s received. At present we can only process v1 to v%s modifiable user data.' % (data_schema_version, cls.CURRENT_SCHEMA_VERSION))
        return cls.from_dict(raw_user_data_dict)

class ExplorationUserDataDict(TypedDict):
    """Dictionary representing the ExplorationUserData object."""
    rating: Optional[int]
    rated_on: Optional[datetime.datetime]
    draft_change_list: Optional[List[Dict[str, str]]]
    draft_change_list_last_updated: Optional[datetime.datetime]
    draft_change_list_exp_version: Optional[int]
    draft_change_list_id: int
    mute_suggestion_notifications: bool
    mute_feedback_notifications: bool
    furthest_reached_checkpoint_exp_version: Optional[int]
    furthest_reached_checkpoint_state_name: Optional[str]
    most_recently_reached_checkpoint_exp_version: Optional[int]
    most_recently_reached_checkpoint_state_name: Optional[str]

class ExplorationUserData:
    """Value object representing a user's exploration data.

    Attributes:
        user_id: str. The user id.
        exploration_id: str. The exploration id.
        rating: int or None. The rating (1-5) the user assigned to the
            exploration.
        rated_on: datetime or None. When the most recent rating was awarded,
            or None if not rated.
        draft_change_list: list(dict) or None. List of uncommitted changes made
            by the user to the exploration.
        draft_change_list_last_updated: datetime or None. Timestamp of when the
            change list was last updated.
        draft_change_list_exp_version: int or None. The exploration version
            that this change list applied to.
        draft_change_list_id: int. The version of the draft change list which
            was last saved by the user.
        mute_suggestion_notifications: bool. The user's preference for
            receiving suggestion emails for this exploration.
        mute_feedback_notifications: bool. The user's preference for receiving
            feedback emails for this exploration.
        furthest_reached_checkpoint_exp_version: int or None. The exploration
            version of furthest reached checkpoint.
        furthest_reached_checkpoint_state_name: str or None. The state name
            of the furthest reached checkpoint.
        most_recently_reached_checkpoint_exp_version: int or None. The
            exploration version of the most recently reached checkpoint.
        most_recently_reached_checkpoint_state_name: str or None. The state
            name of the most recently reached checkpoint.
    """

    def __init__(self, user_id: str, exploration_id: str, rating: Optional[int]=None, rated_on: Optional[datetime.datetime]=None, draft_change_list: Optional[List[Dict[str, str]]]=None, draft_change_list_last_updated: Optional[datetime.datetime]=None, draft_change_list_exp_version: Optional[int]=None, draft_change_list_id: int=0, mute_suggestion_notifications: bool=feconf.DEFAULT_SUGGESTION_NOTIFICATIONS_MUTED_PREFERENCE, mute_feedback_notifications: bool=feconf.DEFAULT_FEEDBACK_NOTIFICATIONS_MUTED_PREFERENCE, furthest_reached_checkpoint_exp_version: Optional[int]=None, furthest_reached_checkpoint_state_name: Optional[str]=None, most_recently_reached_checkpoint_exp_version: Optional[int]=None, most_recently_reached_checkpoint_state_name: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        "Constructs a ExplorationUserData domain object.\n\n        Attributes:\n            user_id: str. The user id.\n            exploration_id: str. The exploration id.\n            rating: int or None. The rating (1-5) the user assigned to the\n                exploration.\n            rated_on: datetime or None. When the most recent rating was\n                awarded, or None if not rated.\n            draft_change_list: list(dict) or None. List of uncommitted\n                changes made by the user to the exploration.\n            draft_change_list_last_updated: datetime or None. Timestamp of\n                when the change list was last updated.\n            draft_change_list_exp_version: int or None. The exploration\n                version that this change list applied to.\n            draft_change_list_id: int. The version of the draft change list\n                which was last saved by the user.\n            mute_suggestion_notifications: bool. The user's preference for\n                receiving suggestion emails for this exploration.\n            mute_feedback_notifications: bool. The user's preference for\n                receiving feedback emails for this exploration.\n            furthest_reached_checkpoint_exp_version: int or None. The\n                exploration version of furthest reached checkpoint.\n            furthest_reached_checkpoint_state_name: str or None. The\n                state name of the furthest reached checkpoint.\n            most_recently_reached_checkpoint_exp_version: int or None. The\n                exploration version of the most recently reached\n                checkpoint.\n            most_recently_reached_checkpoint_state_name: str or None. The\n                state name of the most recently reached checkpoint.\n        "
        self.user_id = user_id
        self.exploration_id = exploration_id
        self.rating = rating
        self.rated_on = rated_on
        self.draft_change_list = draft_change_list
        self.draft_change_list_last_updated = draft_change_list_last_updated
        self.draft_change_list_exp_version = draft_change_list_exp_version
        self.draft_change_list_id = draft_change_list_id
        self.mute_suggestion_notifications = mute_suggestion_notifications
        self.mute_feedback_notifications = mute_feedback_notifications
        self.furthest_reached_checkpoint_exp_version = furthest_reached_checkpoint_exp_version
        self.furthest_reached_checkpoint_state_name = furthest_reached_checkpoint_state_name
        self.most_recently_reached_checkpoint_exp_version = most_recently_reached_checkpoint_exp_version
        self.most_recently_reached_checkpoint_state_name = most_recently_reached_checkpoint_state_name

    def to_dict(self) -> ExplorationUserDataDict:
        if False:
            i = 10
            return i + 15
        'Convert the ExplorationUserData domain instance into a dictionary\n        form with its keys as the attributes of this class.\n\n        Returns:\n            dict. A dictionary containing the UserSettings class information\n            in a dictionary form.\n        '
        return {'rating': self.rating, 'rated_on': self.rated_on, 'draft_change_list': self.draft_change_list, 'draft_change_list_last_updated': self.draft_change_list_last_updated, 'draft_change_list_exp_version': self.draft_change_list_exp_version, 'draft_change_list_id': self.draft_change_list_id, 'mute_suggestion_notifications': self.mute_suggestion_notifications, 'mute_feedback_notifications': self.mute_feedback_notifications, 'furthest_reached_checkpoint_exp_version': self.furthest_reached_checkpoint_exp_version, 'furthest_reached_checkpoint_state_name': self.furthest_reached_checkpoint_state_name, 'most_recently_reached_checkpoint_exp_version': self.most_recently_reached_checkpoint_exp_version, 'most_recently_reached_checkpoint_state_name': self.most_recently_reached_checkpoint_state_name}

class LearnerGroupsUserDict(TypedDict):
    """Dictionary for LearnerGroupsUser domain object."""
    user_id: str
    invited_to_learner_groups_ids: List[str]
    learner_groups_user_details: List[LearnerGroupUserDetailsDict]
    learner_groups_user_details_schema_version: int

class LearnerGroupUserDetailsDict(TypedDict):
    """Dictionary for user details of a particular learner group."""
    group_id: str
    progress_sharing_is_turned_on: bool

class LearnerGroupUserDetails:
    """Domain object for user details of a particular learner group."""

    def __init__(self, group_id: str, progress_sharing_is_turned_on: bool) -> None:
        if False:
            i = 10
            return i + 15
        'Constructs a LearnerGroupUserDetails domain object.\n\n        Attributes:\n            group_id: str. The id of the learner group.\n            progress_sharing_is_turned_on: bool. Whether progress sharing is\n                turned on for the learner group.\n        '
        self.group_id = group_id
        self.progress_sharing_is_turned_on = progress_sharing_is_turned_on

    def to_dict(self) -> LearnerGroupUserDetailsDict:
        if False:
            for i in range(10):
                print('nop')
        'Convert the LearnerGroupUserDetails domain instance into a\n        dictionary form with its keys as the attributes of this class.\n\n        Returns:\n            dict. A dictionary containing the LearnerGroupUserDetails class\n            information in a dictionary form.\n        '
        return {'group_id': self.group_id, 'progress_sharing_is_turned_on': self.progress_sharing_is_turned_on}

class LearnerGroupsUser:
    """Domain object for learner groups user."""

    def __init__(self, user_id: str, invited_to_learner_groups_ids: List[str], learner_groups_user_details: List[LearnerGroupUserDetails], learner_groups_user_details_schema_version: int) -> None:
        if False:
            return 10
        'Constructs a LearnerGroupsUser domain object.\n\n        Attributes:\n            user_id: str. The user id.\n            invited_to_learner_groups_ids: list(str). List of learner group ids\n                that the user has been invited to join as learner.\n            learner_groups_user_details:\n                list(LearnerGroupUserDetails). List of user details of\n                all learner groups that the user is learner of.\n            learner_groups_user_details_schema_version: int. The version\n                of the learner groups user details schema blob.\n        '
        self.user_id = user_id
        self.invited_to_learner_groups_ids = invited_to_learner_groups_ids
        self.learner_groups_user_details = learner_groups_user_details
        self.learner_groups_user_details_schema_version = learner_groups_user_details_schema_version

    def to_dict(self) -> LearnerGroupsUserDict:
        if False:
            return 10
        'Convert the LearnerGroupsUser domain instance into a dictionary\n        form with its keys as the attributes of this class.\n\n        Returns:\n            dict. A dictionary containing the LearnerGroupsUser class\n            information in a dictionary form.\n        '
        learner_groups_user_details_dict = [learner_group_details.to_dict() for learner_group_details in self.learner_groups_user_details]
        return {'user_id': self.user_id, 'invited_to_learner_groups_ids': self.invited_to_learner_groups_ids, 'learner_groups_user_details': learner_groups_user_details_dict, 'learner_groups_user_details_schema_version': self.learner_groups_user_details_schema_version}

    def validate(self) -> None:
        if False:
            return 10
        'Validates the LearnerGroupsUser domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the LearnerGroupsUser\n                are invalid.\n        '
        for learner_group_details in self.learner_groups_user_details:
            if learner_group_details.group_id in self.invited_to_learner_groups_ids:
                raise utils.ValidationError('Learner cannot be invited to join learner group %s since they are already its learner.' % learner_group_details.group_id)

class TranslationCoordinatorStatsDict(TypedDict):
    """Dict representation of TranslationCoordinatorStats domain object."""
    language_id: str
    coordinator_ids: List[str]
    coordinators_count: int

class TranslationCoordinatorStats:
    """Domain object for the TranslationCoordinatorStatsModel."""

    def __init__(self, language_id: str, coordinator_ids: List[str], coordinators_count: int) -> None:
        if False:
            return 10
        self.language_id = language_id
        self.coordinator_ids = coordinator_ids
        self.coordinators_count = coordinators_count

    def to_dict(self) -> TranslationCoordinatorStatsDict:
        if False:
            i = 10
            return i + 15
        'Returns a dict representaion of TranslationCoordinatorStats.\n\n        Returns: dict. The dict representation.\n        '
        return {'language_id': self.language_id, 'coordinator_ids': self.coordinator_ids, 'coordinators_count': self.coordinators_count}