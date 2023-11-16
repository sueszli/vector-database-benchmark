import bitfield.models
from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import F, Q

def reset_is_private_flag(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    UserMessage = apps.get_model('zerver', 'UserMessage')
    UserProfile = apps.get_model('zerver', 'UserProfile')
    user_profile_ids = UserProfile.objects.all().order_by('id').values_list('id', flat=True)
    i = 0
    total = len(user_profile_ids)
    print('Setting default values for the new flag...', flush=True)
    for user_id in user_profile_ids:
        while True:
            flag_set_objects = UserMessage.objects.filter(user_profile_id=user_id).extra(where=['flags & 2048 != 0']).order_by('message_id')[0:1000]
            user_message_ids = flag_set_objects.values_list('id', flat=True)
            count = UserMessage.objects.filter(id__in=user_message_ids).update(flags=F('flags').bitand(~UserMessage.flags.is_private))
            if count < 1000:
                break
        i += 1
        if i % 50 == 0 or i == total:
            percent = round(i / total * 100, 2)
            print(f'Processed {i}/{total} {percent}%', flush=True)

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0176_remove_subscription_notifications')]
    operations = [migrations.AlterField(model_name='archivedusermessage', name='flags', field=bitfield.models.BitField(['read', 'starred', 'collapsed', 'mentioned', 'wildcard_mentioned', 'summarize_in_home', 'summarize_in_stream', 'force_expand', 'force_collapse', 'has_alert_word', 'historical', 'is_private'], default=0)), migrations.AlterField(model_name='usermessage', name='flags', field=bitfield.models.BitField(['read', 'starred', 'collapsed', 'mentioned', 'wildcard_mentioned', 'summarize_in_home', 'summarize_in_stream', 'force_expand', 'force_collapse', 'has_alert_word', 'historical', 'is_private'], default=0)), migrations.AddIndex(model_name='usermessage', index=models.Index('user_profile', 'message', condition=Q(flags__andnz=2048), name='zerver_usermessage_is_private_message_id')), migrations.RunPython(reset_is_private_flag, reverse_code=migrations.RunPython.noop, elidable=True)]