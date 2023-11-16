from django.apps.registry import Apps
from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor

def update_allowed_projects_for_all_paid_subscriptions(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    organisation_subscription_information_cache_model = apps.get_model('organisations', 'organisationsubscriptioninformationcache')
    organisation_subscription_information_cache_model.objects.exclude(organisation__subscription__plan='free').update(allowed_projects=None)

def reverse(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    '\n    Set the values for the OrganisationSubscriptionInformationCache objects back to 1 (which\n    is incorrect for paid subscriptions, but necessary to avoid IntegrityError when reversing\n    migrations)\n    '
    organisation_subscription_information_cache_model = apps.get_model('organisations', 'organisationsubscriptioninformationcache')
    organisation_subscription_information_cache_model.objects.exclude(organisation__subscription__plan='free').update(allowed_projects=1)

class Migration(migrations.Migration):
    dependencies = [('organisations', '0045_auto_20230802_1956')]
    operations = [migrations.AlterField(model_name='organisationsubscriptioninformationcache', name='allowed_projects', field=models.IntegerField(default=1, blank=True, null=True)), migrations.RunPython(update_allowed_projects_for_all_paid_subscriptions, reverse_code=reverse)]