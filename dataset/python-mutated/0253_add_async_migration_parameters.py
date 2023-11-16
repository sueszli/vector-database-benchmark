from django.db import migrations, models

class AddFieldNullSafe(migrations.AddField):

    def describe(self):
        if False:
            for i in range(10):
                print('nop')
        return super().describe() + ' -- not-null-ignore'

class Migration(migrations.Migration):
    dependencies = [('posthog', '0252_reset_insight_refreshing_status')]
    operations = [AddFieldNullSafe(model_name='asyncmigration', name='parameters', field=models.JSONField(default=dict))]