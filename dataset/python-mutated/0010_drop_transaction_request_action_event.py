from django.db import migrations

def drop_transaction_action_request(apps, _schema_editor):
    if False:
        return 10
    WebhookEvent = apps.get_model('webhook', 'WebhookEvent')
    WebhookEvent.objects.filter(event_type='transaction_action_request').delete()

class Migration(migrations.Migration):
    dependencies = [('webhook', '0009_webhook_custom_headers')]
    operations = [migrations.RunPython(drop_transaction_action_request, reverse_code=migrations.RunPython.noop)]