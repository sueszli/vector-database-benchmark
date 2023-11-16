from django.db import migrations, transaction
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import Max, Min
from django.utils.timezone import now as timezone_now

def backfill_missing_subscriptions(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    "Backfill subscription realm audit log events for users which are\n    currently subscribed but don't have any, presumably due to some\n    historical bug.  This is important because those rows are\n    necessary when reactivating a user who is currently\n    soft-deactivated.\n\n    For each stream, we find the subscribed users who have no relevant\n    realm audit log entries, and create a backfill=True subscription\n    audit log entry which is the latest it could have been, based on\n    UserMessage rows.\n\n    "
    Stream = apps.get_model('zerver', 'Stream')
    RealmAuditLog = apps.get_model('zerver', 'RealmAuditLog')
    Subscription = apps.get_model('zerver', 'Subscription')
    UserMessage = apps.get_model('zerver', 'UserMessage')
    Message = apps.get_model('zerver', 'Message')

    def get_last_message_id() -> int:
        if False:
            return 10
        last_id = Message.objects.aggregate(Max('id'))['id__max']
        if last_id is None:
            last_id = -1
        return last_id
    for stream in Stream.objects.all():
        with transaction.atomic():
            subscribed_user_ids = set(Subscription.objects.filter(recipient_id=stream.recipient_id).values_list('user_profile_id', flat=True))
            user_ids_in_audit_log = set(RealmAuditLog.objects.filter(realm=stream.realm, event_type__in=[301, 302, 303], modified_stream=stream).distinct('modified_user_id').values_list('modified_user_id', flat=True))
            user_ids_missing_events = subscribed_user_ids - user_ids_in_audit_log
            if not user_ids_missing_events:
                continue
            last_message_id = get_last_message_id()
            now = timezone_now()
            backfills = []
            for user_id in sorted(user_ids_missing_events):
                print(f'Backfilling subscription event for {user_id} in stream {stream.id} in realm {stream.realm.string_id}')
                aggregated = UserMessage.objects.filter(user_profile_id=user_id, message__recipient=stream.recipient_id).aggregate(earliest_date=Min('message__date_sent'), earliest_message_id=Min('message_id'), latest_date=Max('message__date_sent'), latest_message_id=Max('message_id'))
                if aggregated['earliest_message_id'] is not None:
                    event_last_message_id = aggregated['earliest_message_id'] - 1
                else:
                    event_last_message_id = last_message_id
                if aggregated['earliest_date'] is not None:
                    event_time = aggregated['earliest_date']
                else:
                    event_time = now
                log_event = RealmAuditLog(event_time=event_time, event_last_message_id=event_last_message_id, backfilled=True, event_type=301, realm_id=stream.realm_id, modified_user_id=user_id, modified_stream_id=stream.id)
                backfills.append(log_event)
                sub = Subscription.objects.get(user_profile_id=user_id, recipient_id=stream.recipient_id)
                if sub.active:
                    continue
                if aggregated['latest_message_id'] is not None:
                    event_last_message_id = aggregated['latest_message_id']
                else:
                    event_last_message_id = last_message_id
                if aggregated['latest_date'] is not None:
                    event_time = aggregated['latest_date']
                else:
                    event_time = now
                deactivated_log_event = RealmAuditLog(event_time=event_time, event_last_message_id=event_last_message_id, backfilled=True, event_type=303, realm_id=stream.realm_id, modified_user_id=user_id, modified_stream_id=stream.id)
                backfills.append(deactivated_log_event)
            RealmAuditLog.objects.bulk_create(backfills)

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0449_scheduledmessage_zerver_unsent_scheduled_messages_indexes')]
    operations = [migrations.RunPython(backfill_missing_subscriptions, reverse_code=migrations.RunPython.noop, elidable=True)]