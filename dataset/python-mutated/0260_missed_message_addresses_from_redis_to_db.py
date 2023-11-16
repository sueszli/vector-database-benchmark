import secrets
from django.core.exceptions import ObjectDoesNotExist
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from zerver.lib.redis_utils import get_redis_client

def generate_missed_message_token() -> str:
    if False:
        i = 10
        return i + 15
    return 'mm' + secrets.token_hex(16)

def move_missed_message_addresses_to_database(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        return 10
    redis_client = get_redis_client()
    MissedMessageEmailAddress = apps.get_model('zerver', 'MissedMessageEmailAddress')
    UserProfile = apps.get_model('zerver', 'UserProfile')
    Message = apps.get_model('zerver', 'Message')
    Recipient = apps.get_model('zerver', 'Recipient')
    RECIPIENT_PERSONAL = 1
    RECIPIENT_STREAM = 2
    all_mm_keys = redis_client.keys('missed_message:*')
    for key in all_mm_keys:
        if redis_client.hincrby(key, 'uses_left', -1) < 0:
            redis_client.delete(key)
            continue
        (user_profile_id, recipient_id, subject_b) = redis_client.hmget(key, 'user_profile_id', 'recipient_id', 'subject')
        if user_profile_id is None or recipient_id is None or subject_b is None:
            redis_client.delete(key)
            continue
        topic_name = subject_b.decode()
        try:
            user_profile = UserProfile.objects.get(id=user_profile_id)
            recipient = Recipient.objects.get(id=recipient_id)
            if recipient.type == RECIPIENT_STREAM:
                message = Message.objects.filter(subject__iexact=topic_name, recipient_id=recipient.id).latest('id')
            elif recipient.type == RECIPIENT_PERSONAL:
                message = Message.objects.filter(recipient_id=user_profile.recipient_id, sender_id=recipient.type_id).latest('id')
            else:
                message = Message.objects.filter(recipient_id=recipient.id).latest('id')
        except ObjectDoesNotExist:
            redis_client.delete(key)
            continue
        MissedMessageEmailAddress.objects.create(message=message, user_profile=user_profile, email_token=generate_missed_message_token())
        redis_client.delete(key)

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0259_missedmessageemailaddress')]
    operations = [migrations.RunPython(move_missed_message_addresses_to_database, reverse_code=migrations.RunPython.noop, elidable=True)]