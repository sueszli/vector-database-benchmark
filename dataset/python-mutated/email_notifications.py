"""
Code for generating email notifications for users (e.g. email notifications for
new activities in your dashboard activity stream) and emailing them to the
users.

"""
from __future__ import annotations
import datetime
import re
from typing import Any
from jinja2 import Environment
import ckan.model as model
import ckan.logic as logic
import ckan.lib.jinja_extensions as jinja_extensions
from ckan.common import ungettext, ugettext, config
from ckan.types import Context

def string_to_timedelta(s: str) -> datetime.timedelta:
    if False:
        i = 10
        return i + 15
    'Parse a string s and return a standard datetime.timedelta object.\n\n    Handles days, hours, minutes, seconds, and microseconds.\n\n    Accepts strings in these formats:\n\n    2 days\n    14 days\n    4:35:00 (hours, minutes and seconds)\n    4:35:12.087465 (hours, minutes, seconds and microseconds)\n    7 days, 3:23:34\n    7 days, 3:23:34.087465\n    .087465 (microseconds only)\n\n    :raises ckan.logic.ValidationError: if the given string does not match any\n        of the recognised formats\n\n    '
    patterns = []
    days_only_pattern = '(?P<days>\\d+)\\s+day(s)?'
    patterns.append(days_only_pattern)
    hms_only_pattern = '(?P<hours>\\d?\\d):(?P<minutes>\\d\\d):(?P<seconds>\\d\\d)'
    patterns.append(hms_only_pattern)
    ms_only_pattern = '.(?P<milliseconds>\\d\\d\\d)(?P<microseconds>\\d\\d\\d)'
    patterns.append(ms_only_pattern)
    hms_and_ms_pattern = hms_only_pattern + ms_only_pattern
    patterns.append(hms_and_ms_pattern)
    days_and_hms_pattern = '{0},\\s+{1}'.format(days_only_pattern, hms_only_pattern)
    patterns.append(days_and_hms_pattern)
    days_and_hms_and_ms_pattern = days_and_hms_pattern + ms_only_pattern
    patterns.append(days_and_hms_and_ms_pattern)
    match = None
    for pattern in patterns:
        match = re.match('^{0}$'.format(pattern), s)
        if match:
            break
    if not match:
        raise logic.ValidationError({'message': 'Not a valid time: {0}'.format(s)})
    gd = match.groupdict()
    days = int(gd.get('days', '0'))
    hours = int(gd.get('hours', '0'))
    minutes = int(gd.get('minutes', '0'))
    seconds = int(gd.get('seconds', '0'))
    milliseconds = int(gd.get('milliseconds', '0'))
    microseconds = int(gd.get('microseconds', '0'))
    delta = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds, microseconds=microseconds)
    return delta

def render_activity_email(activities: list[dict[str, Any]]) -> str:
    if False:
        return 10
    globals = {'site_title': config.get('ckan.site_title')}
    template_name = 'activity_streams/activity_stream_email_notifications.text'
    env = Environment(**jinja_extensions.get_jinja_env_options())
    env.install_gettext_callables(ugettext, ungettext)
    template = env.get_template(template_name, globals=globals)
    return template.render({'activities': activities})

def _notifications_for_activities(activities: list[dict[str, Any]], user_dict: dict[str, Any]) -> list[dict[str, str]]:
    if False:
        return 10
    "Return one or more email notifications covering the given activities.\n\n    This function handles grouping multiple activities into a single digest\n    email.\n\n    :param activities: the activities to consider\n    :type activities: list of activity dicts like those returned by\n        ckan.logic.action.get.dashboard_activity_list()\n\n    :returns: a list of email notifications\n    :rtype: list of dicts each with keys 'subject' and 'body'\n\n    "
    if not activities:
        return []
    if not user_dict.get('activity_streams_email_notifications'):
        return []
    subject = ungettext('{n} new activity from {site_title}', '{n} new activities from {site_title}', len(activities)).format(site_title=config.get('ckan.site_title'), n=len(activities))
    body = render_activity_email(activities)
    notifications = [{'subject': subject, 'body': body}]
    return notifications

def _notifications_from_dashboard_activity_list(user_dict: dict[str, Any], since: datetime.datetime) -> list[dict[str, str]]:
    if False:
        return 10
    "Return any email notifications from the given user's dashboard activity\n    list since `since`.\n\n    "
    context: Context = {'user': user_dict['id']}
    activity_list = logic.get_action('dashboard_activity_list')(context, {})
    activity_list = [activity for activity in activity_list if activity['user_id'] != user_dict['id']]
    strptime = datetime.datetime.strptime
    fmt = '%Y-%m-%dT%H:%M:%S.%f'
    activity_list = [activity for activity in activity_list if strptime(activity['timestamp'], fmt) > since]
    return _notifications_for_activities(activity_list, user_dict)
_notifications_functions = [_notifications_from_dashboard_activity_list]

def get_notifications(user_dict: dict[str, Any], since: datetime.datetime) -> list[dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    "Return any email notifications for the given user since `since`.\n\n    For example email notifications about activity streams will be returned for\n    any activities the occurred since `since`.\n\n    :param user_dict: a dictionary representing the user, should contain 'id'\n        and 'name'\n    :type user_dict: dictionary\n\n    :param since: datetime after which to return notifications from\n    :rtype since: datetime.datetime\n\n    :returns: a list of email notifications\n    :rtype: list of dicts with keys 'subject' and 'body'\n\n    "
    notifications = []
    for function in _notifications_functions:
        notifications.extend(function(user_dict, since))
    return notifications

def send_notification(user: dict[str, Any], email_dict: dict[str, Any]) -> None:
    if False:
        while True:
            i = 10
    'Email `email_dict` to `user`.'
    import ckan.lib.mailer
    if not user.get('email'):
        return
    try:
        ckan.lib.mailer.mail_recipient(user['display_name'], user['email'], email_dict['subject'], email_dict['body'])
    except ckan.lib.mailer.MailerException:
        raise

def get_and_send_notifications_for_user(user: dict[str, Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    email_notifications_since = config.get('ckan.email_notifications_since')
    email_notifications_since = string_to_timedelta(email_notifications_since)
    email_notifications_since = datetime.datetime.utcnow() - email_notifications_since
    dashboard = model.Dashboard.get(user['id'])
    if dashboard:
        email_last_sent = dashboard.email_last_sent
        activity_stream_last_viewed = dashboard.activity_stream_last_viewed
        since = max(email_notifications_since, email_last_sent, activity_stream_last_viewed)
        notifications = get_notifications(user, since)
        for notification in notifications:
            send_notification(user, notification)
        dashboard.email_last_sent = datetime.datetime.utcnow()
        model.repo.commit()

def get_and_send_notifications_for_all_users() -> None:
    if False:
        print('Hello World!')
    context: Context = {'ignore_auth': True, 'keep_email': True}
    users = logic.get_action('user_list')(context, {})
    for user in users:
        get_and_send_notifications_for_user(user)