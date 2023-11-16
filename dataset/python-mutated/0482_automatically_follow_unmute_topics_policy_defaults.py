from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import Max
AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_SEND = 2
AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION = 3

def set_default_user_topic_policies(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        return 10
    RealmUserDefault = apps.get_model('zerver', 'RealmUserDefault')
    UserProfile = apps.get_model('zerver', 'UserProfile')
    BATCH_SIZE = 1000
    max_id_realm_user_default = RealmUserDefault.objects.aggregate(Max('id', default=0))['id__max'] + BATCH_SIZE
    max_id_user_profile = UserProfile.objects.aggregate(Max('id', default=0))['id__max'] + BATCH_SIZE
    lower_bound = 0
    while lower_bound < max_id_realm_user_default:
        RealmUserDefault.objects.filter(id__gt=lower_bound, id__lte=lower_bound + BATCH_SIZE).update(automatically_follow_topics_policy=AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, automatically_unmute_topics_in_muted_streams_policy=AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_SEND)
        lower_bound += BATCH_SIZE
    lower_bound = 0
    while lower_bound < max_id_user_profile:
        UserProfile.objects.filter(id__gt=lower_bound, id__lte=lower_bound + BATCH_SIZE).update(automatically_follow_topics_policy=AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION, automatically_unmute_topics_in_muted_streams_policy=AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_SEND)
        lower_bound += BATCH_SIZE

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0481_alter_realm_uuid_alter_realm_uuid_owner_secret')]
    operations = [migrations.RunPython(set_default_user_topic_policies, reverse_code=migrations.RunPython.noop, elidable=True), migrations.AlterField(model_name='userprofile', name='automatically_follow_topics_policy', field=models.PositiveSmallIntegerField(null=False, default=AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION)), migrations.AlterField(model_name='userprofile', name='automatically_unmute_topics_in_muted_streams_policy', field=models.PositiveSmallIntegerField(null=False, default=AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_SEND)), migrations.AlterField(model_name='realmuserdefault', name='automatically_follow_topics_policy', field=models.PositiveSmallIntegerField(null=False, default=AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION)), migrations.AlterField(model_name='realmuserdefault', name='automatically_unmute_topics_in_muted_streams_policy', field=models.PositiveSmallIntegerField(null=False, default=AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_SEND))]