from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
REALM_NAME_IN_EMAIL_NOTIFICATIONS_POLICY_AUTOMATIC = 1
REALM_NAME_IN_EMAIL_NOTIFICATIONS_POLICY_ALWAYS = 2
REALM_NAME_IN_EMAIL_NOTIFICATIONS_POLICY_NEVER = 3

def update_realm_name_in_email_notifications_policy_values(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        for i in range(10):
            print('nop')
    UserProfile = apps.get_model('zerver', 'UserProfile')
    UserProfile.objects.filter(realm_name_in_notifications=True).update(realm_name_in_email_notifications_policy=REALM_NAME_IN_EMAIL_NOTIFICATIONS_POLICY_ALWAYS)

def reverse_code(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        return 10
    UserProfile = apps.get_model('zerver', 'UserProfile')
    UserProfile.objects.filter(realm_name_in_email_notifications_policy=REALM_NAME_IN_EMAIL_NOTIFICATIONS_POLICY_ALWAYS).update(realm_name_in_notifications=True)

def update_realm_name_in_email_notifications_policy_values_for_realm_user_default(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        return 10
    RealmUserDefault = apps.get_model('zerver', 'RealmUserDefault')
    RealmUserDefault.objects.filter(realm_name_in_notifications=True).update(realm_name_in_email_notifications_policy=REALM_NAME_IN_EMAIL_NOTIFICATIONS_POLICY_ALWAYS)

def reverse_code_for_realm_user_default(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    RealmUserDefault = apps.get_model('zerver', 'RealmUserDefault')
    RealmUserDefault.objects.filter(realm_name_in_email_notifications_policy=REALM_NAME_IN_EMAIL_NOTIFICATIONS_POLICY_ALWAYS).update(realm_name_in_notifications=True)

class Migration(migrations.Migration):
    dependencies = [('zerver', '0431_alter_archivedreaction_unique_together_and_more')]
    operations = [migrations.AddField(model_name='realmuserdefault', name='realm_name_in_email_notifications_policy', field=models.PositiveSmallIntegerField(default=REALM_NAME_IN_EMAIL_NOTIFICATIONS_POLICY_AUTOMATIC)), migrations.RunPython(update_realm_name_in_email_notifications_policy_values_for_realm_user_default, reverse_code=reverse_code_for_realm_user_default, elidable=True), migrations.AddField(model_name='userprofile', name='realm_name_in_email_notifications_policy', field=models.PositiveSmallIntegerField(default=REALM_NAME_IN_EMAIL_NOTIFICATIONS_POLICY_AUTOMATIC)), migrations.RunPython(update_realm_name_in_email_notifications_policy_values, reverse_code=reverse_code, elidable=True), migrations.RemoveField(model_name='realmuserdefault', name='realm_name_in_notifications'), migrations.RemoveField(model_name='userprofile', name='realm_name_in_notifications')]