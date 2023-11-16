"""Domain objects for app feedback reports."""
from __future__ import annotations
import datetime
import re
from core import feconf
from core import utils
from core.domain import app_feedback_report_constants
from core.domain import story_domain
from core.domain import topic_domain
from typing import Any, Dict, List, Match, Optional, TypedDict, Union
from core.domain import exp_services
from core.platform import models
MYPY = False
if MYPY:
    from mypy_imports import app_feedback_report_models
(app_feedback_report_models,) = models.Registry.import_models([models.Names.APP_FEEDBACK_REPORT])

class AppFeedbackReportDict(TypedDict):
    """Dictionary representing the AppFeedbackReport object."""
    report_id: str
    schema_version: int
    platform: str
    submitted_on_timestamp: str
    local_timezone_offset_hrs: int
    ticket_id: Optional[str]
    scrubbed_by: Optional[str]
    user_supplied_feedback: UserSuppliedFeedbackDict
    device_system_context: DeviceSystemContextDict
    app_context: AppContextDict

class AndroidFeedbackReportDict(TypedDict):
    """Type for the android feedback report dictionary."""
    platform_type: str
    android_report_info_schema_version: int
    app_context: AndroidAppContextDict
    device_context: AndroidDeviceContextDict
    report_submission_timestamp_sec: int
    report_submission_utc_offset_hrs: int
    system_context: AndroidSystemContextDict
    user_supplied_feedback: UserSuppliedFeedbackDict

class AppFeedbackReport:
    """Domain object for a single feedback report."""

    def __init__(self, report_id: str, schema_version: int, platform: str, submitted_on_timestamp: datetime.datetime, local_timezone_offset_hrs: int, ticket_id: Optional[str], scrubbed_by: Optional[str], user_supplied_feedback: UserSuppliedFeedback, device_system_context: DeviceSystemContext, app_context: AppContext) -> None:
        if False:
            i = 10
            return i + 15
        "Constructs an AppFeedbackReport domain object.\n\n        Args:\n            report_id: str. The unique ID of the report.\n            schema_version: int. The schema version of this feedback report.\n            platform: str. The platform this report is for.\n            submitted_on_timestamp: datetime.datetime. Timestamp in seconds\n                since epoch (in UTC) of when the report was submitted by the\n                user.\n            local_timezone_offset_hrs: int. The number of hours offset from UTC\n                for the user's local time.\n            ticket_id: str|None. The unique ID that this ticket is assigned to;\n                None if this report is not yet ticketed.\n            scrubbed_by: str|None. The unique ID of the user that scrubbed this\n                report, or feconf.REPORT_SCRUBBER_BOT_ID if scrubbed by the\n                cron job.\n            user_supplied_feedback: UserSuppliedFeedback. An object representing\n                the information fileld out by the user in the report.\n            device_system_context: DeviceSystemContext. An object representing\n                the user's device and system information used to submit the\n                report.\n            app_context: AppContext. An object representing the user's Oppia\n                app state when they submitted the report.\n        "
        self.report_id = report_id
        self.schema_version = schema_version
        self.platform = platform
        self.submitted_on_timestamp = submitted_on_timestamp
        self.local_timezone_offset_hrs = local_timezone_offset_hrs
        self.ticket_id = ticket_id
        self.scrubbed_by = scrubbed_by
        self.user_supplied_feedback = user_supplied_feedback
        self.device_system_context = device_system_context
        self.app_context = app_context

    def to_dict(self) -> AppFeedbackReportDict:
        if False:
            while True:
                i = 10
        'Returns a dict representing this AppFeedbackReport domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of AppFeedbackReport instance.\n        '
        return {'report_id': self.report_id, 'schema_version': self.schema_version, 'platform': self.platform, 'submitted_on_timestamp': utils.get_human_readable_time_string(utils.get_time_in_millisecs(self.submitted_on_timestamp)), 'local_timezone_offset_hrs': self.local_timezone_offset_hrs, 'ticket_id': self.ticket_id, 'scrubbed_by': self.scrubbed_by, 'user_supplied_feedback': self.user_supplied_feedback.to_dict(), 'device_system_context': self.device_system_context.to_dict(), 'app_context': self.app_context.to_dict()}

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Validates all properties of this report and its constituents.\n\n        Raises:\n            ValidationError. One or more attributes of the AppFeedbackReport are\n                not valid.\n            NotImplementedError. The full validation for web report domain\n                objects is not implemented yet.\n        '
        if self.platform == app_feedback_report_constants.PLATFORM_CHOICE_WEB:
            raise NotImplementedError('Domain objects for web reports have not been implemented yet.')
        self.require_valid_platform(self.platform)
        self.require_valid_schema_version(self.platform, self.schema_version)
        if self.scrubbed_by is not None:
            self.require_valid_scrubber_id(self.scrubbed_by)
        if not app_feedback_report_constants.TIMEZONE_MINIMUM_OFFSET <= self.local_timezone_offset_hrs <= app_feedback_report_constants.TIMEZONE_MAXIMUM_OFFSET:
            raise utils.ValidationError('Expected local timezone offset to be in [%d, %d], received: %d' % (app_feedback_report_constants.TIMEZONE_MINIMUM_OFFSET, app_feedback_report_constants.TIMEZONE_MAXIMUM_OFFSET, self.local_timezone_offset_hrs))
        if self.ticket_id is not None:
            AppFeedbackReportTicket.require_valid_ticket_id(self.ticket_id)
        self.user_supplied_feedback.validate()
        if not isinstance(self.device_system_context, AndroidDeviceSystemContext):
            raise utils.ValidationError('Expected device and system context to be of type AndroidDeviceSystemContext for platform %s, received: %r' % (self.platform, self.device_system_context.__class__))
        self.device_system_context.validate()
        self.app_context.validate()

    @classmethod
    def require_valid_platform(cls, platform: str) -> None:
        if False:
            print('Hello World!')
        'Checks whether the platform is valid.\n\n        Args:\n            platform: str. The platform to validate.\n\n        Raises:\n            ValidationError. No platform supplied.\n            ValidationError. The platform is not supported.\n        '
        if platform is None:
            raise utils.ValidationError('No platform supplied.')
        if platform not in app_feedback_report_constants.PLATFORM_CHOICES:
            raise utils.ValidationError('Report platform should be one of %s, received: %s' % (app_feedback_report_constants.PLATFORM_CHOICES, platform))

    @classmethod
    def require_valid_schema_version(cls, platform: str, schema_version: int) -> None:
        if False:
            print('Hello World!')
        'Checks whether the report schema version is valid for the given\n        platform.\n\n        Args:\n            platform: str. The platform to validate the schema version for.\n            schema_version: int. The schema version to validate.\n\n        Raises:\n            ValidationError. No schema version supplied.\n            ValidationError. The schema version is not supported.\n        '
        assert platform == app_feedback_report_constants.PLATFORM_CHOICE_ANDROID
        minimum_schema = feconf.MINIMUM_ANDROID_REPORT_SCHEMA_VERSION
        current_schema = feconf.CURRENT_ANDROID_REPORT_SCHEMA_VERSION
        if not isinstance(schema_version, int) or schema_version <= 0:
            raise utils.ValidationError('The report schema version %r is invalid, expected an integer in [%d, %d].' % (schema_version, minimum_schema, current_schema))
        if not minimum_schema <= schema_version <= current_schema:
            raise utils.ValidationError('The supported report schema versions for %s reports are [%d, %d], received: %d.' % (platform, minimum_schema, current_schema, schema_version))

    @classmethod
    def require_valid_scrubber_id(cls, scrubber_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the scrubbed_by user is valid.\n\n        Args:\n            scrubber_id: str. The user id to validate.\n\n        Raises:\n            ValidationError. The user id is not a string.\n            ValidationError. The user id is not a valid id format.\n        '
        if not isinstance(scrubber_id, str):
            raise utils.ValidationError('The scrubbed_by user must be a string, but got %r' % scrubber_id)
        if not utils.is_user_id_valid(scrubber_id) and scrubber_id != feconf.APP_FEEDBACK_REPORT_SCRUBBER_BOT_ID:
            raise utils.ValidationError('The scrubbed_by user id %r is invalid.' % scrubber_id)

    @classmethod
    def from_submitted_feedback_dict(cls, report_dict: AndroidFeedbackReportDict) -> AppFeedbackReport:
        if False:
            print('Hello World!')
        'Returns an AppFeedbackReport object from a dict of the report sent in\n        an incoming feedback report request.\n\n        Args:\n            report_dict: dict. A dict representing the incoming feedback report\n                sent in a request.\n\n        Returns:\n            AppFeedbackReport. The corresponding AppFeedbackReport domain\n            object.\n\n        Raises:\n            NotImplementedError. Domain objects for web reports not implemented.\n        '
        if report_dict['platform_type'] == app_feedback_report_constants.PLATFORM_CHOICE_ANDROID:
            return cls.get_android_report_from_dict(report_dict)
        else:
            raise NotImplementedError('Domain objects for web reports must be implemented.')

    @classmethod
    def get_android_report_from_dict(cls, report_dict: AndroidFeedbackReportDict) -> AppFeedbackReport:
        if False:
            for i in range(10):
                print('nop')
        'Returns an AppFeedbackReport object from a dict for an Android\n        report.\n\n        Args:\n            report_dict: dict. A dict representing the Android feedback report.\n\n        Returns:\n            AppFeedbackReport. The corresponding AppFeedbackReport domain\n            object for an Android report.\n        '
        user_supplied_feedback_json = report_dict['user_supplied_feedback']
        user_supplied_feedback_obj = UserSuppliedFeedback(cls.get_report_type_from_string(user_supplied_feedback_json['report_type']), cls.get_category_from_string(user_supplied_feedback_json['category']), user_supplied_feedback_json['user_feedback_selected_items'], user_supplied_feedback_json['user_feedback_other_text_input'])
        system_context_json = report_dict['system_context']
        device_context_json = report_dict['device_context']
        device_system_context_obj = AndroidDeviceSystemContext(system_context_json['platform_version'], system_context_json['package_version_code'], system_context_json['android_device_country_locale_code'], system_context_json['android_device_language_locale_code'], device_context_json['android_device_model'], device_context_json['android_sdk_version'], device_context_json['build_fingerprint'], cls.get_android_network_type_from_string(device_context_json['network_type']))
        app_context_json = report_dict['app_context']
        entry_point_obj = cls.get_entry_point_from_json(app_context_json['entry_point'])
        app_context_obj = AndroidAppContext(entry_point_obj, app_context_json['text_language_code'], app_context_json['audio_language_code'], cls.get_android_text_size_from_string(app_context_json['text_size']), app_context_json['only_allows_wifi_download_and_update'], app_context_json['automatically_update_topics'], app_context_json['account_is_profile_admin'], app_context_json['event_logs'], app_context_json['logcat_logs'])
        report_datetime = datetime.datetime.fromtimestamp(report_dict['report_submission_timestamp_sec'])
        report_id = app_feedback_report_models.AppFeedbackReportModel.generate_id(report_dict['platform_type'], report_datetime)
        report_obj = AppFeedbackReport(report_id, report_dict['android_report_info_schema_version'], report_dict['platform_type'], report_datetime, report_dict['report_submission_utc_offset_hrs'], None, None, user_supplied_feedback_obj, device_system_context_obj, app_context_obj)
        return report_obj

    @classmethod
    def get_report_type_from_string(cls, report_type_name: str) -> app_feedback_report_constants.ReportType:
        if False:
            while True:
                i = 10
        'Determines the report type based on the JSON value.\n\n        Args:\n            report_type_name: str. The name of the report type.\n\n        Returns:\n            REPORT_TYPE. The enum representing this report type.\n        '
        for report_type in app_feedback_report_constants.ALLOWED_REPORT_TYPES:
            if report_type_name == report_type.value:
                return report_type
        raise utils.InvalidInputException('The given report type %s is invalid.' % report_type_name)

    @classmethod
    def get_category_from_string(cls, category_name: str) -> app_feedback_report_constants.Category:
        if False:
            i = 10
            return i + 15
        'Determines the category based on the JSON value.\n\n        Args:\n            category_name: str. The name of the report type.\n\n        Returns:\n            CATEGORY. The enum representing this category.\n        '
        for category_type in app_feedback_report_constants.ALLOWED_CATEGORIES:
            if category_name == category_type.value:
                return category_type
        raise utils.InvalidInputException('The given category %s is invalid.' % category_name)

    @classmethod
    def get_android_text_size_from_string(cls, text_size_name: str) -> app_feedback_report_constants.AndroidTextSize:
        if False:
            for i in range(10):
                print('nop')
        "Determines the app text size based on the JSON value.\n\n        Args:\n            text_size_name: str. The name of the app's text size set.\n\n        Returns:\n            AndroidTextSize. The enum representing the text size.\n        "
        for text_size_type in app_feedback_report_constants.ALLOWED_ANDROID_TEXT_SIZES:
            if text_size_name == text_size_type.value:
                return text_size_type
        raise utils.InvalidInputException('The given Android app text size %s is invalid.' % text_size_name)

    @classmethod
    def get_entry_point_from_json(cls, entry_point_json: EntryPointDict) -> EntryPoint:
        if False:
            while True:
                i = 10
        'Determines the entry point type based on the received JSON.\n\n        Args:\n            entry_point_json: dict. The JSON data of the entry point.\n\n        Returns:\n            EntryPoint. The EntryPoint domain object representing the entry\n            point.\n\n        Raises:\n            InvalidInputException. The given entry point is invalid.\n            Exception. No topic_id provided for LessonPlayerEntryPoint.\n            Exception. No story_id provided for LessonPlayerEntryPoint.\n            Exception. No exploration_id provided for LessonPlayerEntryPoint.\n            Exception. No topic_id provided for RevisionCardEntryPoint.\n            Exception. No subtopic_id provided for RevisionCardEntryPoint.\n        '
        entry_point_name = entry_point_json['entry_point_name']
        if entry_point_name == app_feedback_report_constants.EntryPoint.NAVIGATION_DRAWER.value:
            return NavigationDrawerEntryPoint()
        elif entry_point_name == app_feedback_report_constants.EntryPoint.LESSON_PLAYER.value:
            if entry_point_json['entry_point_topic_id'] is None:
                raise Exception('No topic_id provided for LessonPlayerEntryPoint.')
            if entry_point_json['entry_point_story_id'] is None:
                raise Exception('No story_id provided for LessonPlayerEntryPoint.')
            if entry_point_json['entry_point_exploration_id'] is None:
                raise Exception('No exploration_id provided for LessonPlayerEntryPoint.')
            return LessonPlayerEntryPoint(entry_point_json['entry_point_topic_id'], entry_point_json['entry_point_story_id'], entry_point_json['entry_point_exploration_id'])
        elif entry_point_name == app_feedback_report_constants.EntryPoint.REVISION_CARD.value:
            if entry_point_json['entry_point_topic_id'] is None:
                raise Exception('No topic_id provided for RevisionCardEntryPoint.')
            if entry_point_json['entry_point_subtopic_id'] is None:
                raise Exception('No subtopic_id provided for RevisionCardEntryPoint.')
            return RevisionCardEntryPoint(entry_point_json['entry_point_topic_id'], entry_point_json['entry_point_subtopic_id'])
        elif entry_point_name == app_feedback_report_constants.EntryPoint.CRASH.value:
            return CrashEntryPoint()
        else:
            raise utils.InvalidInputException('The given entry point %s is invalid.' % entry_point_name)

    @classmethod
    def get_android_network_type_from_string(cls, network_type_name: str) -> app_feedback_report_constants.AndroidNetworkType:
        if False:
            while True:
                i = 10
        'Determines the network type based on the JSON value.\n\n        Args:\n            network_type_name: str. The name of the network type.\n\n        Returns:\n            AndroidNetworkType. The enum representing the network type.\n        '
        for network_type in app_feedback_report_constants.ALLOWED_ANDROID_NETWORK_TYPES:
            if network_type_name == network_type.value:
                return network_type
        raise utils.InvalidInputException('The given Android network type %s is invalid.' % network_type_name)

class UserSuppliedFeedbackDict(TypedDict):
    """Dictionary representing the UserSuppliedFeedback object."""
    report_type: str
    category: str
    user_feedback_selected_items: List[str]
    user_feedback_other_text_input: str

class UserSuppliedFeedback:
    """Domain object for the user-supplied information in feedback reports."""

    def __init__(self, report_type: app_feedback_report_constants.ReportType, category: app_feedback_report_constants.Category, user_feedback_selected_items: List[str], user_feedback_other_text_input: str) -> None:
        if False:
            print('Hello World!')
        'Constructs a UserSuppliedFeedback domain object.\n\n        Args:\n            report_type: ReportType. The type of feedback submitted by the user\n                as an enum.\n            category: Category. The category enum that this specific report_type\n                is providing feedback on that correponds.\n            user_feedback_selected_items: list(str). A list of strings that\n                represent any options selected by the user for the feedback\n                they are providing in this feedback report.\n            user_feedback_other_text_input: str. The open text inputted by\n                the user.\n        '
        self.report_type = report_type
        self.category = category
        self.user_feedback_selected_items = user_feedback_selected_items
        self.user_feedback_other_text_input = user_feedback_other_text_input

    def to_dict(self) -> UserSuppliedFeedbackDict:
        if False:
            return 10
        'Returns a dict representing this UserSuppliedFeedback domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of UserSuppliedFeedback instance.\n        '
        return {'report_type': self.report_type.value, 'category': self.category.value, 'user_feedback_selected_items': self.user_feedback_selected_items, 'user_feedback_other_text_input': self.user_feedback_other_text_input}

    def validate(self) -> None:
        if False:
            return 10
        'Validates this UserSuppliedFeedback domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the UserSuppliedFeedback\n                are not valid.\n        '
        self.require_valid_report_type(self.report_type)
        self.require_valid_category(self.category)
        if self.user_feedback_selected_items is None:
            raise utils.ValidationError('No user_feedback_selected_items supplied.')
        if self.user_feedback_other_text_input is None:
            raise utils.ValidationError('No user_feedback_selected_items supplied for category %s.' % self.category)
        self.require_valid_user_feedback_items_for_category(self.category, self.user_feedback_selected_items, self.user_feedback_other_text_input)

    @classmethod
    def require_valid_report_type(cls, report_type: app_feedback_report_constants.ReportType) -> None:
        if False:
            i = 10
            return i + 15
        'Checks whether the report_type is valid.\n\n        Args:\n            report_type: ReportType. The report type enum to validate.\n\n        Raises:\n            ValidationError. No report_type supplied.\n            ValidationError. The report_type is not supported.\n        '
        if report_type is None:
            raise utils.ValidationError('No report_type supplied.')
        if report_type not in app_feedback_report_constants.ALLOWED_REPORT_TYPES:
            raise utils.ValidationError('Invalid report type %s, must be one of %s.' % (report_type, app_feedback_report_constants.ALLOWED_REPORT_TYPES))

    @classmethod
    def require_valid_category(cls, category: app_feedback_report_constants.Category) -> None:
        if False:
            print('Hello World!')
        'Checks whether the category is valid.\n\n        Args:\n            category: CATEGORY. The category enum to validate.\n\n        Raises:\n            ValidationError. No category supplied.\n            ValidationError. The category is not supported.\n        '
        if category is None:
            raise utils.ValidationError('No category supplied.')
        if category not in app_feedback_report_constants.ALLOWED_CATEGORIES:
            raise utils.ValidationError('Invalid category %s, must be one of %s.' % (category, app_feedback_report_constants.ALLOWED_CATEGORIES))

    @classmethod
    def require_valid_user_feedback_items_for_category(cls, category: app_feedback_report_constants.Category, selected_items: List[str], other_text_input: str) -> None:
        if False:
            return 10
        'Checks whether the user_feedback_selected_items and\n        user_feedback_selected_items are valid for the given cateory and\n        selected items.\n\n        Args:\n            category: str. The category selected for this report.\n            selected_items: list(str). The user feedback selected items to\n                validate, chosen by the user in the report.\n            other_text_input: str. The user feedback other text input to\n                validate, provided by the user in the report.\n\n        Raises:\n            ValidationError. The given selection items and text input for the\n                category are not valid.\n        '
        if category in app_feedback_report_constants.ALLOWED_SELECTION_ITEMS_CATEGORIES:
            cls.require_valid_selected_items_for_category(selected_items)
        if category in app_feedback_report_constants.ALLOWED_ONLY_INPUT_TEXT_CATEGORIES:
            if len(selected_items) != 0:
                raise utils.ValidationError('Report cannot have selection options for category %r.' % category)
            if not isinstance(other_text_input, str):
                raise utils.ValidationError('Invalid input text, must be a string, received: %r.' % other_text_input)

    @classmethod
    def require_valid_selected_items_for_category(cls, selected_items: List[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the user_feedback_selected_items are valid.\n\n        Args:\n            selected_items: list(str). The items selected by the user to\n                validate.\n\n        Raises:\n            ValidationError. The item is not a valid selection option.\n        '
        for item in selected_items:
            if not isinstance(item, str):
                raise utils.ValidationError('Invalid option %s selected by user.' % item)

class DeviceSystemContextDict(TypedDict):
    """Dictionary representing the DeviceSystemContext object."""
    version_name: str
    device_country_locale_code: str

class DeviceSystemContext:
    """Domain object for the device and system information from the device used
    to submit the report.
    """

    def __init__(self, version_name: str, device_country_locale_code: str) -> None:
        if False:
            print('Hello World!')
        "Constructs a DeviceSystemContext domain object.\n\n        Args:\n            version_name: str. The specific version of the app being used to\n                submit the report.\n            device_country_locale_code: str. The user's country locale\n                represented as an ISO-3166 code.\n        "
        self.version_name = version_name
        self.device_country_locale_code = device_country_locale_code

    def to_dict(self) -> DeviceSystemContextDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this DeviceSystemContext domain object.\n        Subclasses should override this to propertly format any additional\n        properties.\n\n        Returns:\n            dict. A dict, mapping all fields of DeviceSystemContext instance.\n        '
        return {'version_name': self.version_name, 'device_country_locale_code': self.device_country_locale_code}

    def validate(self) -> None:
        if False:
            return 10
        'Validates this DeviceSystemContext domain object.\n\n        Raises:\n            NotImplementedError. The derived child classes must implement the\n                necessary logic as described above.\n        '
        raise NotImplementedError('Subclasses of DeviceSystemContext should implement domain validation.')

class AndroidDeviceContextDict(TypedDict):
    """Type for the android device context dictionary."""
    android_device_model: str
    android_sdk_version: int
    build_fingerprint: str
    network_type: str

class AndroidSystemContextDict(TypedDict):
    """Type for the android system context dictionary."""
    platform_version: str
    package_version_code: int
    android_device_country_locale_code: str
    android_device_language_locale_code: str

class AndroidDeviceSystemContextDict(TypedDict):
    """Dictionary representing the AndroidDeviceSystemContext object."""
    version_name: str
    package_version_code: int
    device_country_locale_code: str
    device_language_locale_code: str
    device_model: str
    sdk_version: int
    build_fingerprint: str
    network_type: str

class AndroidDeviceSystemContext(DeviceSystemContext):
    """Domain object for the device and system information specific to an
    Android device.
    """

    def __init__(self, version_name: str, package_version_code: int, device_country_locale_code: str, device_language_locale_code: str, device_model: str, sdk_version: int, build_fingerprint: str, network_type: app_feedback_report_constants.AndroidNetworkType) -> None:
        if False:
            i = 10
            return i + 15
        "Constructs an AndroidDeviceSystemContext domain object.\n\n        Args:\n            version_name: str. The specific version of the app being used to\n                submit the report.\n            package_version_code: int. The Oppia Android package version on the\n                device.\n            device_country_locale_code: str. The device's country locale code\n                as an ISO-639 code, as determined in the Android device's\n                settings.\n            device_language_locale_code: str. The device's language locale code\n                as an ISO-639 code, as determined in the Android device's\n                settings.\n            device_model: str. The Android device model used to send the report.\n            sdk_version: int. The Android SDK version running on the device.\n            build_fingerprint: str. The unique build fingerprint of this app\n                version.\n            network_type: AndroidNetworkType. The enum for the network type\n                the device was connected to.\n        "
        super().__init__(version_name, device_country_locale_code)
        self.package_version_code = package_version_code
        self.device_language_locale_code = device_language_locale_code
        self.device_model = device_model
        self.sdk_version = sdk_version
        self.build_fingerprint = build_fingerprint
        self.network_type = network_type

    def to_dict(self) -> AndroidDeviceSystemContextDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this AndroidDeviceSystemContext domain\n        object.\n\n        Returns:\n            dict. A dict, mapping all fields of AndroidDeviceSystemContext\n            instance.\n        '
        return {'version_name': self.version_name, 'package_version_code': self.package_version_code, 'device_country_locale_code': self.device_country_locale_code, 'device_language_locale_code': self.device_language_locale_code, 'device_model': self.device_model, 'sdk_version': self.sdk_version, 'build_fingerprint': self.build_fingerprint, 'network_type': self.network_type.value}

    def validate(self) -> None:
        if False:
            return 10
        'Validates this AndroidDeviceSystemContext domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                AndroidDeviceSystemContext are not valid.\n        '
        self.require_valid_version_name(self.version_name)
        self.require_valid_package_version_code(self.package_version_code)
        self.require_valid_locale_code('country', self.device_country_locale_code)
        self.require_valid_locale_code('language', self.device_language_locale_code)
        if self.device_model is None:
            raise utils.ValidationError('No device model supplied.')
        if not isinstance(self.device_model, str):
            raise utils.ValidationError('Android device model must be an string, received: %r.' % self.device_model)
        self.require_valid_sdk_version(self.sdk_version)
        if self.build_fingerprint is None:
            raise utils.ValidationError('No build fingerprint supplied.')
        if not isinstance(self.build_fingerprint, str):
            raise utils.ValidationError('Build fingerprint must be a string, received: %r.' % self.build_fingerprint)
        self.require_valid_network_type(self.network_type)

    @classmethod
    def require_valid_version_name(cls, version_name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the version name is a valid string app version for\n        Oppia Android.\n\n        Args:\n            version_name: str. The version name for this report.\n\n        Raises:\n            ValidationError. The given app version name is not valid.\n        '
        if version_name is None:
            raise utils.ValidationError('No version name supplied.')
        if not isinstance(version_name, str):
            raise utils.ValidationError('Version name must be a string, received: %r.' % version_name)
        delimiter = app_feedback_report_constants.ANDROID_VERSION_NAME_DELIMITER
        if len(version_name.split(delimiter)) != 3:
            raise utils.ValidationError('The version name is not a valid string format, received: %s.' % version_name)

    @classmethod
    def require_valid_package_version_code(cls, package_version_code: int) -> None:
        if False:
            return 10
        'Checks whether the package version code is a valid string code for\n        Oppia Android.\n\n        Args:\n            package_version_code: int. The package version code for this report.\n\n        Raises:\n            ValidationError. The given code is not valid.\n        '
        if package_version_code is None:
            raise utils.ValidationError('No package version code supplied.')
        if not isinstance(package_version_code, int):
            raise utils.ValidationError('Package version code must be an int, received: %r.' % package_version_code)
        if package_version_code < feconf.MINIMUM_ANDROID_PACKAGE_VERSION_CODE:
            raise utils.ValidationError('The package version code is not a valid int. The minimum supported version is %d, received: %d.' % (feconf.MINIMUM_ANDROID_PACKAGE_VERSION_CODE, package_version_code))

    @classmethod
    def require_valid_locale_code(cls, locale_type: str, locale_code: str) -> None:
        if False:
            print('Hello World!')
        "Checks whether the device's locale code is a valid  code.\n\n        Args:\n            locale_type: str. The type of locale code to verify; can be either\n                'country' or 'language'.\n            locale_code: str. The device's country locale code\n                that sent the report.\n\n        Raises:\n            ValidationError. The given code is not valid.\n        "
        if locale_code is None:
            raise utils.ValidationError('No device %s locale code supplied.' % locale_type)
        if not isinstance(locale_code, str):
            raise utils.ValidationError("The device's %s locale code must be an string, received: %r." % (locale_type, locale_code))
        if not cls._match_locale_code_string(locale_code):
            raise utils.ValidationError("The device's %s locale code is not a valid string, received: %s." % (locale_type, locale_code))

    @classmethod
    def _match_locale_code_string(cls, code: str) -> Optional[Match[str]]:
        if False:
            while True:
                i = 10
        "Helper that checks whether the given locale code is a valid code.\n\n        Args:\n            code: str. The device's country locale code that sent the report.\n\n        Returns:\n            bool. Whether the given code is valid. Valid codes are alphabetic\n            string that may contain a number of single hyphens.\n        "
        regex_string = '^([a-z]+[-]?[a-z]+)+$'
        return re.compile(regex_string).match(code.lower())

    @classmethod
    def require_valid_sdk_version(cls, sdk_version: int) -> None:
        if False:
            print('Hello World!')
        "Checks that the Android device's SDK version is a positive integer.\n\n        Args:\n            sdk_version: int. The SDK version of the device sending this report.\n\n        Raises:\n            ValidationError. The given SDK version  is not valid.\n        "
        if sdk_version is None:
            raise utils.ValidationError('No SDK version supplied.')
        if not isinstance(sdk_version, int):
            raise utils.ValidationError('SDK version must be an int, received: %r.' % sdk_version)
        if sdk_version < app_feedback_report_constants.MINIMUM_ANDROID_SDK_VERSION:
            raise utils.ValidationError('Invalid SDK version, received: %s.' % sdk_version)

    @classmethod
    def require_valid_network_type(cls, network_type: app_feedback_report_constants.AndroidNetworkType) -> None:
        if False:
            print('Hello World!')
        "Checks that the Android device's network type is valid.\n\n        Args:\n            network_type: AndroidNetworkType. The network type the device\n                was connected to when sending the report, as an enum.\n\n        Raises:\n            ValidationError. The given network is not a string.\n            ValidationError. The given network is not valid.\n        "
        if network_type is None:
            raise utils.ValidationError('No network type supplied.')
        if network_type not in app_feedback_report_constants.ALLOWED_ANDROID_NETWORK_TYPES:
            raise utils.ValidationError('Invalid network type, received: %s.' % network_type)

class AppContextDict(TypedDict):
    """Dictionary representing the AppContext object."""
    entry_point: EntryPointDict
    text_language_code: str
    audio_language_code: str

class AppContext:
    """Domain object for the Oppia app information of the user's Oppia instance
    at the time they submitted the report.
    """

    def __init__(self, entry_point: EntryPoint, text_language_code: str, audio_language_code: str) -> None:
        if False:
            i = 10
            return i + 15
        'Constructs an AppContext domain object.\n\n        Args:\n            entry_point: EntryPoint. An object representing The entry point that\n                the user used to initiate the report.\n            text_language_code: str. The ISO-639 code for the text language set\n                in the app.\n            audio_language_code: str. The ISO-639 code for the audio language\n                set in the app.\n        '
        self.entry_point = entry_point
        self.text_language_code = text_language_code
        self.audio_language_code = audio_language_code

    def to_dict(self) -> AppContextDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this AppContext domain object. Subclasses\n        should override this to propertly format any additional properties.\n\n        Returns:\n            dict. A dict, mapping all fields of AppContext instance.\n        '
        return {'entry_point': self.entry_point.to_dict(), 'text_language_code': self.text_language_code, 'audio_language_code': self.audio_language_code}

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Validates this AppContext domain object.\n\n        Raises:\n            NotImplementedError. Subclasses should implement their own\n                validation checks.\n        '
        raise NotImplementedError('Subclasses of AppContext should implement their own validation checks.')

class AndroidAppContextDict(TypedDict):
    """Dictionary representing the AndroidAppContext object."""
    entry_point: EntryPointDict
    text_language_code: str
    audio_language_code: str
    text_size: str
    only_allows_wifi_download_and_update: bool
    automatically_update_topics: bool
    account_is_profile_admin: bool
    event_logs: List[str]
    logcat_logs: List[str]

class AndroidAppContext(AppContext):
    """Domain object for the app context information specific to the Oppia
    Android app.
    """

    def __init__(self, entry_point: EntryPoint, text_language_code: str, audio_language_code: str, text_size: app_feedback_report_constants.AndroidTextSize, only_allows_wifi_download_and_update: bool, automatically_update_topics: bool, account_is_profile_admin: bool, event_logs: List[str], logcat_logs: List[str]) -> None:
        if False:
            while True:
                i = 10
        'Constructs a AndroidAppContext domain object.\n\n        Args:\n            entry_point: EntryPoint. An object representing The entry point that\n                the user used to initiate the report.\n            text_language_code: str. The ISO-639 code for the text language set\n                in the app.\n            audio_language_code: str. The ISO-639 code for the audio language\n                set in the app.\n            text_size: AndroidTextSize. The enum type for text size set by\n                the user in the app.\n            only_allows_wifi_download_and_update: bool. True if the user only\n                allows downloads and updates when connected to wifi.\n            automatically_update_topics: bool. True if the user allows\n                automatically updating topics.\n            account_is_profile_admin: bool. True if user sending the report is\n                an admin account.\n            event_logs: list(str). A list of strings for the event logs\n                collected in the app; the list is empty if this instance has\n                been scrubbed.\n            logcat_logs: list(str). A list of strings for the logcat events\n                recorded in the app; the list is empty if this instance has been\n                scrubbed.\n        '
        super().__init__(entry_point, text_language_code, audio_language_code)
        self.entry_point = entry_point
        self.text_language_code = text_language_code
        self.audio_language_code = audio_language_code
        self.text_size = text_size
        self.only_allows_wifi_download_and_update = only_allows_wifi_download_and_update
        self.automatically_update_topics = automatically_update_topics
        self.account_is_profile_admin = account_is_profile_admin
        self.event_logs = event_logs
        self.logcat_logs = logcat_logs

    def to_dict(self) -> AndroidAppContextDict:
        if False:
            print('Hello World!')
        'Returns a dict representing this AndroidAppContext domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of AndroidAppContext instance.\n        '
        return {'entry_point': self.entry_point.to_dict(), 'text_language_code': self.text_language_code, 'audio_language_code': self.audio_language_code, 'text_size': self.text_size.value, 'only_allows_wifi_download_and_update': self.only_allows_wifi_download_and_update, 'automatically_update_topics': self.automatically_update_topics, 'account_is_profile_admin': self.account_is_profile_admin, 'event_logs': self.event_logs, 'logcat_logs': self.logcat_logs}

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Validates this AndroidAppContext domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                AndroidAppContext are not valid.\n        '
        self.entry_point.validate()
        self.require_valid_language_code('text', self.text_language_code)
        self.require_valid_language_code('audio', self.audio_language_code)
        self.require_valid_text_size(self.text_size)
        if self.only_allows_wifi_download_and_update is None or not isinstance(self.only_allows_wifi_download_and_update, bool):
            raise utils.ValidationError('only_allows_wifi_download_and_update field should be a boolean, received: %r' % self.only_allows_wifi_download_and_update)
        if self.automatically_update_topics is None or not isinstance(self.automatically_update_topics, bool):
            raise utils.ValidationError('automatically_update_topics field should be a boolean, received: %r' % self.automatically_update_topics)
        if self.account_is_profile_admin is None or not isinstance(self.account_is_profile_admin, bool):
            raise utils.ValidationError('account_is_profile_admin field should be a boolean, received: %r' % self.account_is_profile_admin)
        if self.event_logs is None or not isinstance(self.event_logs, list):
            raise utils.ValidationError('Should have an event log list, received: %r' % self.event_logs)
        if self.logcat_logs is None or not isinstance(self.logcat_logs, list):
            raise utils.ValidationError('Should have a logcat log list, received: %r' % self.logcat_logs)

    @classmethod
    def require_valid_language_code(cls, language_type: str, language_code: str) -> None:
        if False:
            return 10
        "Checks that the language code is valid.\n\n        Args:\n            language_type: str. The type of language code being validates,\n                either 'text' or 'audio'.\n            language_code: str. The language code being validated, as determined\n                by the Oppia app.\n\n        Raises:\n            ValidationError. The given code is not valid.\n        "
        if language_code is None:
            raise utils.ValidationError('No app %s language code supplied.' % language_type)
        if not isinstance(language_code, str):
            raise utils.ValidationError("Expected the app's %s language code to be a string, received: %r" % (language_type, language_code))
        if not cls._match_language_code_string(language_code):
            raise utils.ValidationError("The app's %s language code is not a valid string, received: %s." % (language_type, language_code))

    @classmethod
    def _match_language_code_string(cls, code: str) -> Optional[Match[str]]:
        if False:
            while True:
                i = 10
        'Helper that checks whether the given language code is a valid code.\n\n        Args:\n            code: str. The language code set on the app.\n\n        Returns:\n            bool. Whether the given code is valid. Valid codes are alphabetic\n            string that may contain a number of single hyphens.\n        '
        regex_string = '^([a-z]+[-]?[a-z]+)+$'
        return re.compile(regex_string).match(code)

    @classmethod
    def require_valid_text_size(cls, text_size: app_feedback_report_constants.AndroidTextSize) -> None:
        if False:
            while True:
                i = 10
        'Checks whether the package version code is a valid string code for\n        Oppia Android.\n\n        Args:\n            text_size: AndroidTextSize. The enum type for the text size set by\n                the user in the app.\n\n        Raises:\n            ValidationError. The given text size is not valid.\n        '
        if text_size is None:
            raise utils.ValidationError('No text size supplied.')
        if text_size not in app_feedback_report_constants.ALLOWED_ANDROID_TEXT_SIZES:
            raise utils.ValidationError('App text size should be one of %s, received: %s' % (app_feedback_report_constants.ALLOWED_ANDROID_TEXT_SIZES, text_size))

class EntryPointDict(TypedDict):
    """Dictionary representing the EntryPoint object."""
    entry_point_name: str
    entry_point_exploration_id: Optional[str]
    entry_point_story_id: Optional[str]
    entry_point_topic_id: Optional[str]
    entry_point_subtopic_id: Optional[str]

class EntryPoint:
    """Domain object for the entry point used to initiate the feedback
    report.
    """

    def __init__(self, entry_point: app_feedback_report_constants.EntryPoint, topic_id: Optional[str]=None, story_id: Optional[str]=None, exploration_id: Optional[str]=None, subtopic_id: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Constructs an EntryPoint domain object.\n\n        Args:\n            entry_point: ENTRY_POINT. The enum type for entry point used.\n            topic_id: str|None. The id for the current topic if the report was\n                sent during a topic in a lesson or revision session.\n            story_id: str|None. The id for the current story if the report was\n                sent during a lesson.\n            exploration_id: str|None. The id for the current exploration if the\n                report was sent during a lesson.\n            subtopic_id: int|None. The id for the current subtopic if the report\n                was sent during a revision session.\n        '
        self.entry_point_name = entry_point.value
        self.topic_id = topic_id
        self.story_id = story_id
        self.exploration_id = exploration_id
        self.subtopic_id = subtopic_id

    def to_dict(self) -> Any:
        if False:
            return 10
        'Returns a dict representing this NavigationDrawerEntryPoint domain\n        object.\n\n        Raises:\n            NotImplementedError. Subclasses should implement their own dict\n                representations.\n        '
        raise NotImplementedError('Subclasses of EntryPoint should implement their own dict representations.')

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates the EntryPoint domain object.\n\n        Raises:\n            NotImplementedError. Subclasses should implement their own\n                validation checks.\n        '
        raise NotImplementedError('Subclasses of EntryPoint should implement their own validation checks.')

    @classmethod
    def require_valid_entry_point_name(cls, actual_name: str, expected_entry_point: app_feedback_report_constants.EntryPoint) -> None:
        if False:
            print('Hello World!')
        'Validates this EntryPoint name.\n\n        Args:\n            actual_name: str. The name used for this entry point object.\n            expected_entry_point: ENTRY_POINT. The enum type that should match\n                the given entry_point_name.\n\n        Raises:\n            ValidationError. The name is not valid for the type.\n        '
        expected_name = expected_entry_point.value
        if actual_name is None:
            raise utils.ValidationError('No entry point name supplied.')
        if not isinstance(actual_name, str):
            raise utils.ValidationError('Entry point name must be a string, received: %r.' % actual_name)
        if actual_name is not expected_name:
            raise utils.ValidationError('Expected entry point name %s, received: %s.' % (expected_name, actual_name))

    @classmethod
    def require_valid_entry_point_exploration(cls, exploration_id: Optional[str], story_id: Optional[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the exploration id is a valid one.\n\n        Args:\n            exploration_id: str. The exploraiton ID to validate.\n            story_id: str. The ID of the story that has this exploration.\n\n        Raises:\n            ValidationError. The exploration ID is not a valid ID.\n        '
        if not isinstance(exploration_id, str):
            raise utils.ValidationError('Exploration id should be a string, received: %r' % exploration_id)
        expected_story_id = exp_services.get_story_id_linked_to_exploration(exploration_id)
        if expected_story_id != story_id:
            raise utils.ValidationError('Exploration with id %s is not part of story with id of %s, should be found in story with id of %s' % (exploration_id, story_id, expected_story_id))

class NavigationDrawerEntryPointDict(TypedDict):
    """Dictionary representing the NavigationDrawerEntryPoint object."""
    entry_point_name: str

class NavigationDrawerEntryPoint(EntryPoint):
    """Domain object for the Android navigation drawer entry point."""

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        'Constructs an NavigationDrawerEntryPoint domain object.'
        super().__init__(app_feedback_report_constants.EntryPoint.NAVIGATION_DRAWER, None, None, None, None)

    def to_dict(self) -> NavigationDrawerEntryPointDict:
        if False:
            i = 10
            return i + 15
        'Returns a dict representing this NavigationDrawerEntryPoint domain\n        object.\n\n        Returns:\n            dict. A dict, mapping all fields of NavigationDrawerEntryPoint\n            instance.\n        '
        return {'entry_point_name': self.entry_point_name}

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Validates this NavigationDrawerEntryPoint domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                NavigationDrawerEntryPoint are not valid.\n        '
        self.require_valid_entry_point_name(self.entry_point_name, app_feedback_report_constants.EntryPoint.NAVIGATION_DRAWER)

class LessonPlayerEntryPointDict(TypedDict):
    """Dictionary representing the LessonPlayerEntryPoint object."""
    entry_point_name: str
    topic_id: Optional[str]
    story_id: Optional[str]
    exploration_id: Optional[str]

class LessonPlayerEntryPoint(EntryPoint):
    """Domain object for the lesson player entry point."""

    def __init__(self, topic_id: str, story_id: str, exploration_id: str) -> None:
        if False:
            while True:
                i = 10
        'Constructs an LessonPlayerEntryPoint domain object.\n\n        Args:\n            topic_id: str. The unique ID for the current topic the user is\n                playing when intiating the report.\n            story_id: str. The unique ID for the current story the user is\n                playing when intiating the report.\n            exploration_id: str. The unique ID for the current exploration the\n                user is playing when intiating the report.\n        '
        super().__init__(app_feedback_report_constants.EntryPoint.LESSON_PLAYER, topic_id=topic_id, story_id=story_id, exploration_id=exploration_id)

    def to_dict(self) -> LessonPlayerEntryPointDict:
        if False:
            print('Hello World!')
        'Returns a dict representing this LessonPlayerEntryPoint domain\n        object.\n\n        Returns:\n            dict. A dict, mapping all fields of LessonPlayerEntryPoint instance.\n        '
        return {'entry_point_name': self.entry_point_name, 'topic_id': self.topic_id, 'story_id': self.story_id, 'exploration_id': self.exploration_id}

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates this LessonPlayerEntryPoint domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                LessonPlayerEntryPoint are not valid.\n        '
        self.require_valid_entry_point_name(self.entry_point_name, app_feedback_report_constants.EntryPoint.LESSON_PLAYER)
        topic_domain.Topic.require_valid_topic_id(self.topic_id)
        if self.story_id is None:
            raise utils.ValidationError('The story_id must be a string value, received None')
        story_domain.Story.require_valid_story_id(self.story_id)
        self.require_valid_entry_point_exploration(self.exploration_id, self.story_id)

class RevisionCardEntryPointDict(TypedDict):
    """Dictionary representing the RevisionCardEntryPoint object."""
    entry_point_name: str
    topic_id: Optional[str]
    subtopic_id: Optional[str]

class RevisionCardEntryPoint(EntryPoint):
    """Domain object for the Android revision card entry point."""

    def __init__(self, topic_id: str, subtopic_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Constructs an RevisionCardEntryPoint domain object.\n\n        Args:\n            topic_id: str. The unique ID for the current topic the user is\n                reviewing when intiating the report.\n            subtopic_id: int. The ID for the current subtopic the user is\n                reviewing when intiating the report.\n        '
        super().__init__(app_feedback_report_constants.EntryPoint.REVISION_CARD, topic_id, None, None, subtopic_id)

    def to_dict(self) -> RevisionCardEntryPointDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this RevisionCardEntryPoint domain\n        object.\n\n        Returns:\n            dict. A dict, mapping all fields of RevisionCardEntryPoint\n            instance.\n        '
        return {'entry_point_name': self.entry_point_name, 'topic_id': self.topic_id, 'subtopic_id': self.subtopic_id}

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Validates this RevisionCardEntryPoint domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                RevisionCardEntryPoint are not valid.\n        '
        self.require_valid_entry_point_name(self.entry_point_name, app_feedback_report_constants.EntryPoint.REVISION_CARD)
        topic_domain.Topic.require_valid_topic_id(self.topic_id)
        if not isinstance(self.subtopic_id, int):
            raise utils.ValidationError('Expected subtopic id to be an int, received %s' % self.subtopic_id)

class CrashEntryPointDict(TypedDict):
    """Dictionary representing the CrashEntryPoint object."""
    entry_point_name: str

class CrashEntryPoint(EntryPoint):
    """Domain object for the Android crash dialog entry point."""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        'Constructs an CrashEntryPoint domain object.'
        super().__init__(app_feedback_report_constants.EntryPoint.CRASH, None, None, None, None)

    def to_dict(self) -> CrashEntryPointDict:
        if False:
            print('Hello World!')
        'Returns a dict representing this CrashEntryPoint domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of CrashEntryPoint\n            instance.\n        '
        return {'entry_point_name': self.entry_point_name}

    def validate(self) -> None:
        if False:
            return 10
        'Validates this CrashEntryPoint domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                CrashEntryPoint are not valid.\n        '
        self.require_valid_entry_point_name(self.entry_point_name, app_feedback_report_constants.EntryPoint.CRASH)

class AppFeedbackReportTicketDict(TypedDict):
    """Dictionary representing the AppFeedbackReportTicket object."""
    ticket_id: str
    ticket_name: str
    platform: str
    github_issue_repo_name: Optional[str]
    github_issue_number: Optional[int]
    archived: bool
    newest_report_creation_timestamp_isoformat: Optional[str]
    reports: List[str]

class AppFeedbackReportTicket:
    """Domain object for a single ticket created for feedback reports."""

    def __init__(self, ticket_id: str, ticket_name: str, platform: str, github_issue_repo_name: Optional[str], github_issue_number: Optional[int], archived: bool, newest_report_creation_timestamp: Optional[datetime.datetime], reports: List[str]) -> None:
        if False:
            print('Hello World!')
        'Constructs a AppFeedbackReportTicket domain object.\n\n        Args:\n            ticket_id: str. The unique ID of the ticket.\n            ticket_name: str. The user-readable name given to this ticket.\n            platform: str. The platform that the reports in this ticket apply\n                to; must be one of PLATFORM_CHOICES.\n            github_issue_repo_name: str|None. The Github repository that has the\n                issue addressing this ticket.\n            github_issue_number: int|None. The Github issue number addressing\n                this ticket.\n            archived: bool. Whether this ticket is archived.\n            newest_report_creation_timestamp: datetime.datetime. Timestamp in\n                UTC of the newest submitted report that is in this ticket, or\n                None if there is no report.\n            reports: list(str). The list of IDs for the AppFeedbackReports\n                assigned to this ticket.\n        '
        self.ticket_id = ticket_id
        self.ticket_name = ticket_name
        self.platform = platform
        self.github_issue_repo_name = github_issue_repo_name
        self.github_issue_number = github_issue_number
        self.archived = archived
        self.newest_report_creation_timestamp = newest_report_creation_timestamp
        self.reports = reports

    def to_dict(self) -> AppFeedbackReportTicketDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this AppFeedbackReportTicket domain\n        object.\n\n        Returns:\n            dict. A dict, mapping all fields of AppFeedbackReportTicket\n            instance.\n        '
        return {'ticket_id': self.ticket_id, 'ticket_name': self.ticket_name, 'platform': self.platform, 'github_issue_repo_name': self.github_issue_repo_name, 'github_issue_number': self.github_issue_number, 'archived': self.archived, 'newest_report_creation_timestamp_isoformat': self.newest_report_creation_timestamp.isoformat() if self.newest_report_creation_timestamp else None, 'reports': self.reports}

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates this AppFeedbackReportTicket domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                AppFeedbackReportTicket are not valid.\n        '
        self.require_valid_ticket_id(self.ticket_id)
        self.require_valid_ticket_name(self.ticket_name)
        AppFeedbackReport.require_valid_platform(self.platform)
        if self.github_issue_repo_name is not None:
            self.require_valid_github_repo(self.github_issue_repo_name)
        if self.github_issue_number is not None:
            if not isinstance(self.github_issue_number, int) or self.github_issue_number < 1:
                raise utils.ValidationError('The Github issue number name must be a positive integer, received: %r' % self.github_issue_number)
        if not isinstance(self.archived, bool):
            raise utils.ValidationError('The ticket archived status must be a boolean, received: %r' % self.archived)
        self.require_valid_report_ids(self.reports)

    @classmethod
    def require_valid_ticket_id(cls, ticket_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Checks whether the ticket id is a valid one.\n\n        Args:\n            ticket_id: str. The ticket id to validate.\n\n        Raises:\n            ValidationError. The id is an invalid format.\n        '
        if not isinstance(ticket_id, str):
            raise utils.ValidationError('The ticket id should be a string, received: %s' % ticket_id)
        if len(ticket_id.split(app_feedback_report_constants.TICKET_ID_DELIMITER)) != 3:
            raise utils.ValidationError('The ticket id %s is invalid' % ticket_id)

    @classmethod
    def require_valid_ticket_name(cls, ticket_name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the ticket name is a valid one.\n\n        Args:\n            ticket_name: str. The ticket name to validate.\n\n        Raises:\n            ValidationError. The name is an invalid format.\n        '
        if ticket_name is None:
            raise utils.ValidationError('No ticket name supplied.')
        if not isinstance(ticket_name, str):
            raise utils.ValidationError('The ticket name should be a string, received: %s' % ticket_name)
        if len(ticket_name) > app_feedback_report_constants.MAXIMUM_TICKET_NAME_LENGTH:
            raise utils.ValidationError('The ticket name is too long, has %d characters but only allowed %d characters' % (len(ticket_name), app_feedback_report_constants.MAXIMUM_TICKET_NAME_LENGTH))

    @classmethod
    def require_valid_report_ids(cls, report_ids: List[str]) -> None:
        if False:
            print('Hello World!')
        'Checks whether the reports in this ticket are valid.\n\n        Args:\n            report_ids: list(str). The list of reports IDs of the reports\n                associated with this ticket.\n\n        Raises:\n            ValidationError. The list of reports is invalid.\n        '
        if report_ids is None:
            raise utils.ValidationError('No reports list supplied.')
        if not isinstance(report_ids, list):
            raise utils.ValidationError('The reports list should be a list, received: %r' % report_ids)
        for report_id in report_ids:
            if app_feedback_report_models.AppFeedbackReportModel.get_by_id(report_id) is None:
                raise utils.ValidationError('The report with id %s is invalid.' % report_id)

    @classmethod
    def require_valid_github_repo(cls, repo_name: str) -> None:
        if False:
            i = 10
            return i + 15
        'Checks whether the reports in this ticket are valid.\n\n        Args:\n            repo_name: str. The name of the repo associated with the Github\n                issue.\n\n        Raises:\n            ValidationError. The repo name is invalid.\n        '
        if not isinstance(repo_name, str):
            raise utils.ValidationError('The Github repo name should be a string, received: %s' % repo_name)
        if repo_name not in app_feedback_report_constants.GITHUB_REPO_CHOICES:
            raise utils.ValidationError('The Github repo %s is invalid, must be one of %s.' % (repo_name, app_feedback_report_constants.GITHUB_REPO_CHOICES))
AcceptableEntryPointClasses = Union[NavigationDrawerEntryPointDict, LessonPlayerEntryPointDict, RevisionCardEntryPointDict, CrashEntryPointDict]

class AppFeedbackReportDailyStatsDict(TypedDict):
    """Dictionary representing the AppFeedbackReportDailyStats object."""
    stats_id: str
    ticket: AppFeedbackReportTicketDict
    platform: str
    stats_tracking_date: str
    total_reports_submitted: int
    daily_param_stats: Dict[str, Dict[str, int]]

class AppFeedbackReportDailyStats:
    """Domain object for report statistics on a single day for a specific
    ticket.
    """

    def __init__(self, stats_id: str, ticket: AppFeedbackReportTicket, platform: str, stats_tracking_date: datetime.date, total_reports_submitted: int, daily_param_stats: Dict[str, ReportStatsParameterValueCounts]) -> None:
        if False:
            i = 10
            return i + 15
        'Constructs a AppFeedbackReportDailyStats domain object.\n\n        Args:\n            stats_id: str. The unique ID for ths stats instance.\n            ticket: AppFeedbackReportTicket. The AppFeedbackReportTicket domain\n                object associated with this ticket.\n            platform: str. The platform these report stats are aggregating for.\n            stats_tracking_date: datetime.date. The date that this object is\n                aggregating stats on, in UTC.\n            total_reports_submitted: int. The total number of reports submitted\n                on this date for this ticket.\n            daily_param_stats: dict. A dict representing the statistics on this\n                date. Keys in this dict correpond to STATS_PARAMETER_NAMES\n                enums, while values are ReportStatsParameterValueCounts objects.\n        '
        self.stats_id = stats_id
        self.ticket = ticket
        self.platform = platform
        self.stats_tracking_date = stats_tracking_date
        self.total_reports_submitted = total_reports_submitted
        self.daily_param_stats = daily_param_stats

    def to_dict(self) -> AppFeedbackReportDailyStatsDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this AppFeedbackReportDailyStats domain\n        object.\n\n        Returns:\n            dict. A dict, mapping all fields of AppFeedbackReportDailyStats\n            instance.\n        '
        return {'stats_id': self.stats_id, 'ticket': self.ticket.to_dict(), 'platform': self.platform, 'stats_tracking_date': self.stats_tracking_date.isoformat(), 'total_reports_submitted': self.total_reports_submitted, 'daily_param_stats': {param_name: param_counts_obj.to_dict() for (param_name, param_counts_obj) in self.daily_param_stats.items()}}

    def validate(self) -> None:
        if False:
            return 10
        'Validates this AppFeedbackReportDailyStats domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                AppFeedbackReportDailyStats are not valid.\n        '
        self.require_valid_stats_id(self.stats_id)
        self.ticket.validate()
        AppFeedbackReport.require_valid_platform(self.platform)
        if not isinstance(self.total_reports_submitted, int):
            raise utils.ValidationError('The total number of submitted reports should be an int, received: %r' % self.total_reports_submitted)
        if self.total_reports_submitted < 0:
            raise utils.ValidationError('The total number of submitted reports should be a non-negative int, received: %d' % self.total_reports_submitted)
        self.require_valid_daily_param_stats(self.daily_param_stats)

    @classmethod
    def require_valid_stats_id(cls, stats_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Checks whether the stats id is a valid one.\n\n        Args:\n            stats_id: str. The stats id to validate.\n\n        Raises:\n            ValidationError. The id is an invalid format.\n        '
        if not isinstance(stats_id, str):
            raise utils.ValidationError('The stats id should be a string, received: %r' % stats_id)
        if len(stats_id.split(app_feedback_report_constants.STATS_ID_DELIMITER)) != 3:
            raise utils.ValidationError('The stats id %s is invalid' % stats_id)

    @classmethod
    def require_valid_daily_param_stats(cls, param_stats: Dict[str, ReportStatsParameterValueCounts]) -> None:
        if False:
            return 10
        'Checks whether the statistics in this domain object are valid.\n\n        Args:\n            param_stats: dict. The dict representing the daily stats for this\n                ticket.\n\n        Raises:\n            ValidationError. The dict is an invalid format.\n        '
        if not isinstance(param_stats, dict):
            raise utils.ValidationError('The parameter stats should be a dict, received: %r' % param_stats)
        allowed_parameter_names = [parameter.value for parameter in app_feedback_report_constants.ALLOWED_STATS_PARAMETERS]
        for (param_name, param_count_obj) in param_stats.items():
            if param_name not in allowed_parameter_names:
                raise utils.ValidationError('The parameter %s is not a valid parameter to aggregate stats on, must be one of %s' % (param_name, allowed_parameter_names))
            param_count_obj.validate()

class ReportStatsParameterValueCounts:
    """Domain object for the number of reports that satisfy a specific parameter
    value.
    """

    def __init__(self, parameter_value_counts: Dict[str, int]) -> None:
        if False:
            return 10
        'Constructs a ReportStatsParameterValueCounts domain object.\n\n        Args:\n            parameter_value_counts: dict. A dict with keys that correpond to a\n                specific value for a given parameter, and integer values for the\n                number of reports that satisfy that value.\n        '
        self.parameter_value_counts = parameter_value_counts

    def to_dict(self) -> Dict[str, int]:
        if False:
            while True:
                i = 10
        'Returns a dict representing this ReportStatsParameterValueCounts\n        domain object.\n\n        Returns:\n            dict. A dict, mapping all fields of ReportStatsParameterValueCounts\n            instance.\n        '
        return self.parameter_value_counts

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates this ReportStatsParameterValueCounts domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                ReportStatsParameterValueCounts are not valid.\n        '
        for (param_value, param_count) in self.parameter_value_counts.items():
            if not isinstance(param_value, str):
                raise utils.ValidationError('The parameter value should be a string, received: %r' % param_value)
            if not isinstance(param_count, int) or param_count < 0:
                raise utils.ValidationError('The parameter value count should be a non-negative int, received: %r' % param_count)

class AppFeedbackReportFilterDict(TypedDict):
    """Dictionary representing the AppFeedbackReportFilter object."""
    filter_field: str
    filter_options: List[str]

class AppFeedbackReportFilter:
    """Domain object for a filter that can be applied to the collection of
    feedback reports.
    """

    def __init__(self, filter_field: app_feedback_report_constants.FilterFieldNames, filter_options: List[str]) -> None:
        if False:
            print('Hello World!')
        'Constructs a AppFeedbackReportFilter domain object.\n\n        Args:\n            filter_field: FilterFieldNames. The enum type for the filter\n                category, correponding to a field in the AppFeedbackReport\n                object.\n            filter_options: list(str). The possible values for the given filter.\n        '
        self.filter_field = filter_field
        self.filter_options = filter_options

    def to_dict(self) -> AppFeedbackReportFilterDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representing this AppFeedbackReportFilter domain\n        object.\n\n        Returns:\n            dict. A dict, mapping all fields of AppFeedbackReportFilter\n            instance.\n        '
        self.filter_options.sort()
        return {'filter_field': self.filter_field.name, 'filter_options': self.filter_options}

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates this AppFeedbackReportFilter domain object.\n\n        Raises:\n            ValidationError. One or more attributes of the\n                AppFeedbackReportFilter are not valid.\n        '
        if self.filter_field not in app_feedback_report_constants.ALLOWED_FILTERS:
            raise utils.ValidationError('The filter field should be one of %s, received: %s' % ([item.name for item in app_feedback_report_constants.ALLOWED_FILTERS], self.filter_field.name))
        if not isinstance(self.filter_options, list):
            raise utils.ValidationError('The filter options should be a list, received: %r' % self.filter_options)