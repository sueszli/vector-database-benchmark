"""Services to operate on app feedback report app_feedback_report_models."""
from __future__ import annotations
import datetime
from core import feconf
from core import utils
from core.domain import app_feedback_report_constants
from core.domain import app_feedback_report_domain
from core.platform import models
from typing import Dict, List, Literal, Optional, Sequence, cast, overload
MYPY = False
if MYPY:
    from mypy_imports import app_feedback_report_models
    from mypy_imports import transaction_services
(app_feedback_report_models,) = models.Registry.import_models([models.Names.APP_FEEDBACK_REPORT])
transaction_services = models.Registry.import_transaction_services()
PLATFORM_ANDROID = app_feedback_report_constants.PLATFORM_CHOICE_ANDROID
PLATFORM_WEB = app_feedback_report_constants.PLATFORM_CHOICE_WEB

@overload
def get_report_models(report_ids: List[str], *, strict: Literal[True]) -> List[app_feedback_report_models.AppFeedbackReportModel]:
    if False:
        while True:
            i = 10
    ...

@overload
def get_report_models(report_ids: List[str]) -> List[Optional[app_feedback_report_models.AppFeedbackReportModel]]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_report_models(report_ids: List[str], *, strict: Literal[False]) -> List[Optional[app_feedback_report_models.AppFeedbackReportModel]]:
    if False:
        i = 10
        return i + 15
    ...

def get_report_models(report_ids: List[str], strict: bool=False) -> Sequence[Optional[app_feedback_report_models.AppFeedbackReportModel]]:
    if False:
        return 10
    'Fetches and returns the AppFeedbackReportModels with the given ids.\n\n    Args:\n        report_ids: list(str). The ids for the models to fetch.\n        strict: bool. Whether to fail noisily if no report model with the given\n            ids exists in the datastore.\n\n    Returns:\n        list(AppFeedbackReportModel). A list of models that correspond to the\n        requested reports.\n\n    Raises:\n        Exception. No AppFeedbackReportModel exists for the given id.\n    '
    report_models = app_feedback_report_models.AppFeedbackReportModel.get_multi(report_ids)
    if strict:
        for (index, report_model) in enumerate(report_models):
            if report_model is None:
                raise Exception('No AppFeedbackReportModel exists for the id %s' % report_ids[index])
    return report_models

def create_report_from_json(report_json: app_feedback_report_domain.AndroidFeedbackReportDict) -> app_feedback_report_domain.AppFeedbackReport:
    if False:
        return 10
    'Creates an AppFeedbackReport domain object instance from the incoming\n    JSON request.\n\n    Args:\n        report_json: dict. The JSON for the app feedback report.\n\n    Returns:\n        AppFeedbackReport. The domain object for an Android feedback report.\n    '
    return app_feedback_report_domain.AppFeedbackReport.from_submitted_feedback_dict(report_json)

def store_incoming_report_stats(report_obj: app_feedback_report_domain.AppFeedbackReport) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Adds a new report's stats to the aggregate stats model.\n\n    Args:\n        report_obj: AppFeedbackReport. AppFeedbackReport domain object.\n\n    Raises:\n        NotImplementedError. Stats aggregation for the domain object\n            have not been implemented yet.\n    "
    if report_obj.platform == PLATFORM_WEB:
        raise NotImplementedError('Stats aggregation for incoming web reports have not been implemented yet.')
    platform = PLATFORM_ANDROID
    unticketed_id = app_feedback_report_constants.UNTICKETED_ANDROID_REPORTS_STATS_TICKET_ID
    all_reports_id = app_feedback_report_constants.ALL_ANDROID_REPORTS_STATS_TICKET_ID
    stats_date = report_obj.submitted_on_timestamp.date()
    _update_report_stats_model_in_transaction(unticketed_id, platform, stats_date, report_obj, 1)
    _update_report_stats_model_in_transaction(all_reports_id, platform, stats_date, report_obj, 1)

@transaction_services.run_in_transaction_wrapper
def _update_report_stats_model_in_transaction(ticket_id: str, platform: str, date: datetime.datetime, report_obj: app_feedback_report_domain.AppFeedbackReport, delta: int) -> None:
    if False:
        return 10
    "Adds a new report's stats to the stats model for a specific ticket's\n    stats. Note that this currently only supports Android reports.\n\n    Args:\n        ticket_id: str. The id of the ticket that we want to update stats for.\n        platform: str. The platform of the report being aggregated.\n        date: datetime.date. The date of the stats.\n        report_obj: AppFeedbackReport. AppFeedbackReport domain object.\n        delta: int. The amount to increment the stats by, depending on if the\n            report is added or removed from the model.\n    "
    report_type = report_obj.user_supplied_feedback.report_type.value
    country_locale_code = report_obj.device_system_context.device_country_locale_code
    entry_point_name = report_obj.app_context.entry_point.entry_point_name
    text_language_code = report_obj.app_context.text_language_code
    audio_language_code = report_obj.app_context.audio_language_code
    report_obj.device_system_context.__class__ = app_feedback_report_domain.AndroidDeviceSystemContext
    android_device_system_context = cast(app_feedback_report_domain.AndroidDeviceSystemContext, report_obj.device_system_context)
    sdk_version = str(android_device_system_context.sdk_version)
    version_name = android_device_system_context.version_name
    stats_id = app_feedback_report_models.AppFeedbackReportStatsModel.calculate_id(platform, ticket_id, date)
    stats_model = app_feedback_report_models.AppFeedbackReportStatsModel.get_by_id(stats_id)
    stats_parameter_names = app_feedback_report_constants.StatsParameterNames
    if stats_model is None:
        assert delta > 0
        stats_dict = {stats_parameter_names.REPORT_TYPE.value: {report_type: 1}, stats_parameter_names.COUNTRY_LOCALE_CODE.value: {country_locale_code: 1}, stats_parameter_names.ENTRY_POINT_NAME.value: {entry_point_name: 1}, stats_parameter_names.TEXT_LANGUAGE_CODE.value: {text_language_code: 1}, stats_parameter_names.AUDIO_LANGUAGE_CODE.value: {audio_language_code: 1}, stats_parameter_names.ANDROID_SDK_VERSION.value: {sdk_version: 1}, stats_parameter_names.VERSION_NAME.value: {version_name: 1}}
        app_feedback_report_models.AppFeedbackReportStatsModel.create(stats_id, platform, ticket_id, date, 0, stats_dict)
        stats_model = app_feedback_report_models.AppFeedbackReportStatsModel.get_by_id(stats_id)
    else:
        stats_dict = stats_model.daily_param_stats
        stats_dict[stats_parameter_names.REPORT_TYPE.value] = calculate_new_stats_count_for_parameter(stats_dict[stats_parameter_names.REPORT_TYPE.value], report_type, delta)
        stats_dict[stats_parameter_names.COUNTRY_LOCALE_CODE.value] = calculate_new_stats_count_for_parameter(stats_dict[stats_parameter_names.COUNTRY_LOCALE_CODE.value], country_locale_code, delta)
        stats_dict[stats_parameter_names.ENTRY_POINT_NAME.value] = calculate_new_stats_count_for_parameter(stats_dict[stats_parameter_names.ENTRY_POINT_NAME.value], entry_point_name, delta)
        stats_dict[stats_parameter_names.AUDIO_LANGUAGE_CODE.value] = calculate_new_stats_count_for_parameter(stats_dict[stats_parameter_names.AUDIO_LANGUAGE_CODE.value], audio_language_code, delta)
        stats_dict[stats_parameter_names.TEXT_LANGUAGE_CODE.value] = calculate_new_stats_count_for_parameter(stats_dict[stats_parameter_names.TEXT_LANGUAGE_CODE.value], text_language_code, delta)
        stats_dict[stats_parameter_names.ANDROID_SDK_VERSION.value] = calculate_new_stats_count_for_parameter(stats_dict[stats_parameter_names.ANDROID_SDK_VERSION.value], sdk_version, delta)
        stats_dict[stats_parameter_names.VERSION_NAME.value] = calculate_new_stats_count_for_parameter(stats_dict[stats_parameter_names.VERSION_NAME.value], version_name, delta)
    stats_model.daily_param_stats = stats_dict
    stats_model.total_reports_submitted += delta
    stats_model.update_timestamps()
    stats_model.put()

def calculate_new_stats_count_for_parameter(current_stats_map: Dict[str, int], current_value: str, delta: int) -> Dict[str, int]:
    if False:
        print('Hello World!')
    'Helper to increment or initialize the stats count for a parameter.\n\n    Args:\n        current_stats_map: dict. The current stats map for the parameter we are\n            updating; keys correspond to the possible value for a single\n            parameter.\n        current_value: str. The value for the parameter that we are updating\n            the stats of.\n        delta: int. The amount to increment the current count by, either -1 or\n            +1.\n\n    Returns:\n        dict. The new stats values for the given parameter.\n    '
    if current_value in current_stats_map:
        current_stats_map[current_value] += delta
    else:
        if delta < 0:
            raise utils.InvalidInputException('Cannot decrement a count for a parameter value that does not exist for this stats model.')
        current_stats_map[current_value] = 1
    return current_stats_map

def get_report_from_model(report_model: app_feedback_report_models.AppFeedbackReportModel) -> app_feedback_report_domain.AppFeedbackReport:
    if False:
        for i in range(10):
            print('nop')
    'Create and return a domain object AppFeedbackReport given a model loaded\n    from the the data.\n\n    Args:\n        report_model: AppFeedbackReportModel. The model loaded from the\n            datastore.\n\n    Returns:\n        AppFeedbackReport. An AppFeedbackReport domain object corresponding to\n        the given model.\n\n    Raises:\n        NotImplementedError. The web report domain object needs to be\n            implemented.\n    '
    if report_model.platform == PLATFORM_ANDROID:
        return get_android_report_from_model(report_model)
    else:
        raise NotImplementedError('Web app feedback report domain objects must be defined.')

def get_ticket_from_model(ticket_model: app_feedback_report_models.AppFeedbackReportTicketModel) -> app_feedback_report_domain.AppFeedbackReportTicket:
    if False:
        print('Hello World!')
    'Create and return a domain object AppFeedbackReportTicket given a model\n    loaded from the the data.\n\n    Args:\n        ticket_model: AppFeedbackReportTicketModel. The model loaded from the\n            datastore.\n\n    Returns:\n        AppFeedbackReportTicket. An AppFeedbackReportTicket domain object\n        corresponding to the given model.\n    '
    return app_feedback_report_domain.AppFeedbackReportTicket(ticket_model.id, ticket_model.ticket_name, ticket_model.platform, ticket_model.github_issue_repo_name, ticket_model.github_issue_number, ticket_model.archived, ticket_model.newest_report_timestamp, ticket_model.report_ids)

def get_stats_from_model(stats_model: app_feedback_report_models.AppFeedbackReportStatsModel) -> app_feedback_report_domain.AppFeedbackReportDailyStats:
    if False:
        print('Hello World!')
    'Create and return a domain object AppFeedbackReportDailyStats given a\n    model loaded from the the storage.\n\n    Args:\n        stats_model: AppFeedbackReportStatsModel. The model loaded from the\n            datastore.\n\n    Returns:\n        AppFeedbackReportDailyStats. An AppFeedbackReportDailyStats domain\n        object corresponding tothe given model.\n    '
    ticket_model = app_feedback_report_models.AppFeedbackReportTicketModel.get_by_id(stats_model.ticket_id)
    ticket_obj = get_ticket_from_model(ticket_model)
    param_stats = create_app_daily_stats_from_model_json(stats_model.daily_param_stats)
    return app_feedback_report_domain.AppFeedbackReportDailyStats(stats_model.id, ticket_obj, stats_model.platform, stats_model.stats_tracking_date, stats_model.total_reports_submitted, param_stats)

def create_app_daily_stats_from_model_json(daily_param_stats: Dict[str, Dict[str, int]]) -> Dict[str, app_feedback_report_domain.ReportStatsParameterValueCounts]:
    if False:
        print('Hello World!')
    "Create and return a dict representing the AppFeedbackReportDailyStats\n    domain object's daily_param_stats.\n\n    Args:\n        daily_param_stats: dict. The stats data from the model.\n\n    Returns:\n        dict. A dict mapping param field names to\n        ReportStatsParameterValueCounts domain objects.\n    "
    stats_dict = {}
    for (stats_name, stats_values_dict) in daily_param_stats.items():
        counts_obj = app_feedback_report_domain.ReportStatsParameterValueCounts(stats_values_dict)
        stats_dict[stats_name] = counts_obj
    return stats_dict

def get_android_report_from_model(android_report_model: app_feedback_report_models.AppFeedbackReportModel) -> app_feedback_report_domain.AppFeedbackReport:
    if False:
        print('Hello World!')
    'Creates a domain object that represents an Android feedback report from\n    the given model.\n\n    Args:\n        android_report_model: AppFeedbackReportModel. The model to convert to a\n            domain object.\n\n    Returns:\n        AppFeedbackReport. The corresponding AppFeedbackReport domain object.\n\n    Raises:\n        NotImplementedError. Android app feedback report migrations not added\n            for new report schemas to be implemented.\n    '
    feedback_report = app_feedback_report_domain.AppFeedbackReport
    if android_report_model.android_report_info_schema_version < feconf.CURRENT_ANDROID_REPORT_SCHEMA_VERSION:
        raise NotImplementedError('Android app feedback report migrations must be added for new report schemas implemented.')
    report_info_dict = android_report_model.android_report_info
    user_supplied_feedback = app_feedback_report_domain.UserSuppliedFeedback(feedback_report.get_report_type_from_string(android_report_model.report_type), feedback_report.get_category_from_string(android_report_model.category), report_info_dict['user_feedback_selected_items'], report_info_dict['user_feedback_other_text_input'])
    device_system_context = app_feedback_report_domain.AndroidDeviceSystemContext(android_report_model.platform_version, report_info_dict['package_version_code'], android_report_model.android_device_country_locale_code, report_info_dict['android_device_language_locale_code'], android_report_model.android_device_model, android_report_model.android_sdk_version, report_info_dict['build_fingerprint'], feedback_report.get_android_network_type_from_string(report_info_dict['network_type']))
    entry_point = feedback_report.get_entry_point_from_json({'entry_point_name': android_report_model.entry_point, 'entry_point_topic_id': android_report_model.entry_point_topic_id, 'entry_point_story_id': android_report_model.entry_point_story_id, 'entry_point_exploration_id': android_report_model.entry_point_exploration_id, 'entry_point_subtopic_id': android_report_model.entry_point_subtopic_id})
    app_context = app_feedback_report_domain.AndroidAppContext(entry_point, android_report_model.text_language_code, android_report_model.audio_language_code, feedback_report.get_android_text_size_from_string(report_info_dict['text_size']), report_info_dict['only_allows_wifi_download_and_update'], report_info_dict['automatically_update_topics'], report_info_dict['account_is_profile_admin'], report_info_dict['event_logs'], report_info_dict['logcat_logs'])
    return app_feedback_report_domain.AppFeedbackReport(android_report_model.id, android_report_model.android_report_info_schema_version, android_report_model.platform, android_report_model.submitted_on, android_report_model.local_timezone_offset_hrs, android_report_model.ticket_id, android_report_model.scrubbed_by, user_supplied_feedback, device_system_context, app_context)

def scrub_all_unscrubbed_expiring_reports(scrubbed_by: str) -> None:
    if False:
        print('Hello World!')
    'Fetches the reports that are expiring and must be scrubbed.\n\n    Args:\n        scrubbed_by: str. The ID of the user initiating scrubbing or\n            feconf.APP_FEEDBACK_REPORT_SCRUBBER_BOT_ID if scrubbed by the cron\n            job.\n    '
    reports_to_scrub = get_all_expiring_reports_to_scrub()
    for report in reports_to_scrub:
        scrub_single_app_feedback_report(report, scrubbed_by)

def get_all_expiring_reports_to_scrub() -> List[app_feedback_report_domain.AppFeedbackReport]:
    if False:
        i = 10
        return i + 15
    'Fetches the reports that are expiring and must be scrubbed.\n\n    Returns:\n        list(AppFeedbackReport). The list of AppFeedbackReportModel domain\n        objects that need to be scrubbed.\n    '
    model_class = app_feedback_report_models.AppFeedbackReportModel
    model_entities = model_class.get_all_unscrubbed_expiring_report_models()
    return [get_report_from_model(model_entity) for model_entity in model_entities]

def scrub_single_app_feedback_report(report: app_feedback_report_domain.AppFeedbackReport, scrubbed_by: str) -> None:
    if False:
        return 10
    'Scrubs the instance of AppFeedbackReportModel with given ID, removing\n    any user-entered input in the entity.\n\n    Args:\n        report: AppFeedbackReport. The domain object of the report to scrub.\n        scrubbed_by: str. The id of the user that is initiating scrubbing of\n            this report, or a constant\n            feconf.APP_FEEDBACK_REPORT_SCRUBBER_BOT_ID if scrubbed by the cron\n            job.\n    '
    report.scrubbed_by = scrubbed_by
    report.user_supplied_feedback.user_feedback_other_text_input = ''
    if report.platform == PLATFORM_ANDROID:
        report.app_context = cast(app_feedback_report_domain.AndroidAppContext, report.app_context)
        report.app_context.event_logs = []
        report.app_context.logcat_logs = []
    save_feedback_report_to_storage(report)

def save_feedback_report_to_storage(report: app_feedback_report_domain.AppFeedbackReport, new_incoming_report: bool=False) -> None:
    if False:
        return 10
    'Saves the AppFeedbackReport domain object to persistent storage.\n\n    Args:\n        report: AppFeedbackReport. The domain object of the report to save.\n        new_incoming_report: bool. Whether the report is a new incoming report\n            that does not have a corresponding model entity.\n    '
    if report.platform == PLATFORM_WEB:
        raise utils.InvalidInputException('Web report domain objects have not been defined.')
    report.validate()
    user_supplied_feedback = report.user_supplied_feedback
    device_system_context = cast(app_feedback_report_domain.AndroidDeviceSystemContext, report.device_system_context)
    app_context = cast(app_feedback_report_domain.AndroidAppContext, report.app_context)
    entry_point = app_context.entry_point
    report_info_json = {'user_feedback_selected_items': user_supplied_feedback.user_feedback_selected_items, 'user_feedback_other_text_input': user_supplied_feedback.user_feedback_other_text_input}
    report_info_json = {'user_feedback_selected_items': user_supplied_feedback.user_feedback_selected_items, 'user_feedback_other_text_input': user_supplied_feedback.user_feedback_other_text_input, 'event_logs': app_context.event_logs, 'logcat_logs': app_context.logcat_logs, 'package_version_code': str(device_system_context.package_version_code), 'android_device_language_locale_code': device_system_context.device_language_locale_code, 'build_fingerprint': device_system_context.build_fingerprint, 'network_type': device_system_context.network_type.value, 'text_size': app_context.text_size.value, 'only_allows_wifi_download_and_update': str(app_context.only_allows_wifi_download_and_update), 'automatically_update_topics': str(app_context.automatically_update_topics), 'account_is_profile_admin': str(app_context.account_is_profile_admin)}
    if new_incoming_report:
        app_feedback_report_models.AppFeedbackReportModel.create(report.report_id, report.platform, report.submitted_on_timestamp, report.local_timezone_offset_hrs, user_supplied_feedback.report_type.value, user_supplied_feedback.category.value, device_system_context.version_name, device_system_context.device_country_locale_code, device_system_context.sdk_version, device_system_context.device_model, entry_point.entry_point_name, entry_point.topic_id, entry_point.story_id, entry_point.exploration_id, entry_point.subtopic_id, app_context.text_language_code, app_context.audio_language_code, None, None)
    model_entity = app_feedback_report_models.AppFeedbackReportModel.get_by_id(report.report_id)
    model_entity.android_report_info = report_info_json
    model_entity.ticket_id = report.ticket_id
    model_entity.scrubbed_by = report.scrubbed_by
    model_entity.update_timestamps()
    model_entity.put()

def get_all_filter_options() -> List[app_feedback_report_domain.AppFeedbackReportFilter]:
    if False:
        i = 10
        return i + 15
    'Fetches all the possible values that moderators can filter reports or\n    tickets by.\n\n    Returns:\n        list(AppFeedbackReportFilter). A list of filters and the possible values\n        they can have.\n    '
    filter_list = []
    model_class = app_feedback_report_models.AppFeedbackReportModel
    for filter_field in app_feedback_report_constants.ALLOWED_FILTERS:
        filter_values = model_class.get_filter_options_for_field(filter_field)
        filter_list.append(app_feedback_report_domain.AppFeedbackReportFilter(filter_field, filter_values))
    return filter_list

def reassign_ticket(report: app_feedback_report_domain.AppFeedbackReport, new_ticket: Optional[app_feedback_report_domain.AppFeedbackReportTicket]) -> None:
    if False:
        return 10
    'Reassign the ticket the report is associated with.\n\n    Args:\n        report: AppFeedbackReport. The report being assigned to a new ticket.\n        new_ticket: AppFeedbackReportTicket|None. The ticket domain object to\n            reassign the report to or None if removing the report form a ticket\n            wihtout reassigning.\n\n    Raises:\n        NotImplementedError. Assigning web reports to tickets has not been\n            implemented.\n    '
    if report.platform == PLATFORM_WEB:
        raise NotImplementedError('Assigning web reports to tickets has not been implemented yet.')
    platform = report.platform
    stats_date = report.submitted_on_timestamp.date()
    old_ticket_id = report.ticket_id
    if old_ticket_id is None:
        _update_report_stats_model_in_transaction(app_feedback_report_constants.UNTICKETED_ANDROID_REPORTS_STATS_TICKET_ID, platform, stats_date, report, -1)
    else:
        old_ticket_model = app_feedback_report_models.AppFeedbackReportTicketModel.get_by_id(old_ticket_id)
        if old_ticket_model is None:
            raise utils.InvalidInputException('The report is being removed from an invalid ticket id: %s.' % old_ticket_id)
        old_ticket_obj = get_ticket_from_model(old_ticket_model)
        old_ticket_obj.reports.remove(report.report_id)
        if len(old_ticket_obj.reports) == 0:
            old_ticket_obj.newest_report_creation_timestamp = None
        elif old_ticket_obj.newest_report_creation_timestamp == report.submitted_on_timestamp:
            report_models = get_report_models(old_ticket_obj.reports, strict=True)
            latest_timestamp = report_models[0].submitted_on
            for index in range(1, len(report_models)):
                if report_models[index].submitted_on > latest_timestamp:
                    latest_timestamp = report_models[index].submitted_on
            old_ticket_obj.newest_report_creation_timestamp = latest_timestamp
        _save_ticket(old_ticket_obj)
        _update_report_stats_model_in_transaction(old_ticket_id, platform, stats_date, report, -1)
    new_ticket_id = app_feedback_report_constants.UNTICKETED_ANDROID_REPORTS_STATS_TICKET_ID
    if new_ticket is not None:
        new_ticket_id = new_ticket.ticket_id
    new_ticket_model = app_feedback_report_models.AppFeedbackReportTicketModel.get_by_id(new_ticket_id)
    new_ticket_obj = get_ticket_from_model(new_ticket_model)
    new_ticket_obj.reports.append(report.report_id)
    if new_ticket_obj.newest_report_creation_timestamp and report.submitted_on_timestamp > new_ticket_obj.newest_report_creation_timestamp:
        new_ticket_obj.newest_report_creation_timestamp = report.submitted_on_timestamp
    _save_ticket(new_ticket_obj)
    platform = report.platform
    stats_date = report.submitted_on_timestamp.date()
    _update_report_stats_model_in_transaction(new_ticket_id, platform, stats_date, report, 1)
    report.ticket_id = new_ticket_id
    save_feedback_report_to_storage(report)

def edit_ticket_name(ticket: app_feedback_report_domain.AppFeedbackReportTicket, new_name: str) -> None:
    if False:
        print('Hello World!')
    'Updates the ticket name.\n\n    Returns:\n        ticket: AppFeedbackReportTicket. The domain object for a ticket.\n        new_name: str. The new name to assign the ticket.\n    '
    ticket.ticket_name = new_name
    _save_ticket(ticket)

def _save_ticket(ticket: app_feedback_report_domain.AppFeedbackReportTicket) -> None:
    if False:
        print('Hello World!')
    'Saves the ticket to persistent storage.\n\n    Returns:\n        ticket: AppFeedbackReportTicket. The domain object to save to storage.\n    '
    model_class = app_feedback_report_models.AppFeedbackReportTicketModel
    ticket_model = model_class.get_by_id(ticket.ticket_id)
    ticket_model.ticket_name = ticket.ticket_name
    ticket_model.platform = ticket.platform
    ticket_model.github_issue_repo_name = ticket.github_issue_repo_name
    ticket_model.github_issue_number = ticket.github_issue_number
    ticket_model.archived = ticket.archived
    ticket_model.newest_report_timestamp = ticket.newest_report_creation_timestamp
    ticket_model.report_ids = ticket.reports
    ticket_model.update_timestamps()
    ticket_model.put()