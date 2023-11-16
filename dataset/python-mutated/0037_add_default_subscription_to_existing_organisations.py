from django.db import migrations

def create_default_subscription(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Organisation = apps.get_model('organisations', 'Organisation')
    Subscription = apps.get_model('organisations', 'Subscription')
    organisations_without_subscription = Organisation.objects.filter(subscription__isnull=True)
    subscriptions_to_create = []
    for organisation in organisations_without_subscription:
        subscriptions_to_create.append(Subscription(organisation=organisation))
    Subscription.objects.bulk_create(subscriptions_to_create)

class Migration(migrations.Migration):
    dependencies = [('organisations', '0036_alter_subscription_plan')]
    operations = [migrations.RunPython(create_default_subscription, reverse_code=migrations.RunPython.noop)]