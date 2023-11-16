from datetime import timedelta
from django.conf import settings
from django.db import transaction
from django.http import HttpRequest, HttpResponse
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from analytics.models import RealmCount
from zerver.actions.realm_export import do_delete_realm_export, notify_realm_export
from zerver.decorator import require_realm_admin
from zerver.lib.exceptions import JsonableError
from zerver.lib.export import get_realm_exports_serialized
from zerver.lib.queue import queue_json_publish
from zerver.lib.response import json_success
from zerver.models import RealmAuditLog, UserProfile

@transaction.atomic(durable=True)
@require_realm_admin
def export_realm(request: HttpRequest, user: UserProfile) -> HttpResponse:
    if False:
        while True:
            i = 10
    event_type = RealmAuditLog.REALM_EXPORTED
    event_time = timezone_now()
    realm = user.realm
    EXPORT_LIMIT = 5
    MAX_MESSAGE_HISTORY = 250000
    MAX_UPLOAD_QUOTA = 10 * 1024 * 1024 * 1024
    event_time_delta = event_time - timedelta(days=7)
    limit_check = RealmAuditLog.objects.filter(realm=realm, event_type=event_type, event_time__gte=event_time_delta)
    if len(limit_check) >= EXPORT_LIMIT:
        raise JsonableError(_('Exceeded rate limit.'))
    exportable_messages_estimate = sum((realm_count.value for realm_count in RealmCount.objects.filter(realm=realm, property='messages_sent:message_type:day', subgroup='public_stream')))
    if exportable_messages_estimate > MAX_MESSAGE_HISTORY or user.realm.currently_used_upload_space_bytes() > MAX_UPLOAD_QUOTA:
        raise JsonableError(_('Please request a manual export from {email}.').format(email=settings.ZULIP_ADMINISTRATOR))
    row = RealmAuditLog.objects.create(realm=realm, event_type=event_type, event_time=event_time, acting_user=user)
    notify_realm_export(user)
    event = {'type': 'realm_export', 'time': event_time, 'realm_id': realm.id, 'user_profile_id': user.id, 'id': row.id}
    transaction.on_commit(lambda : queue_json_publish('deferred_work', event))
    return json_success(request, data={'id': row.id})

@require_realm_admin
def get_realm_exports(request: HttpRequest, user: UserProfile) -> HttpResponse:
    if False:
        for i in range(10):
            print('nop')
    realm_exports = get_realm_exports_serialized(user)
    return json_success(request, data={'exports': realm_exports})

@require_realm_admin
def delete_realm_export(request: HttpRequest, user: UserProfile, export_id: int) -> HttpResponse:
    if False:
        i = 10
        return i + 15
    try:
        audit_log_entry = RealmAuditLog.objects.get(id=export_id, realm=user.realm, event_type=RealmAuditLog.REALM_EXPORTED)
    except RealmAuditLog.DoesNotExist:
        raise JsonableError(_('Invalid data export ID'))
    export_data = audit_log_entry.extra_data
    if export_data.get('deleted_timestamp') is not None:
        raise JsonableError(_('Export already deleted'))
    if export_data.get('export_path') is None:
        if export_data.get('failed_timestamp') is not None:
            raise JsonableError(_('Export failed, nothing to delete'))
        raise JsonableError(_('Export still in progress'))
    do_delete_realm_export(user, audit_log_entry)
    return json_success(request)