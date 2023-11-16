from django.db import migrations

def set_default_currency_for_transaction_event(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    TransactionItem = apps.get_model('payment', 'TransactionItem')
    TransactionEvent = apps.get_model('payment', 'TransactionEvent')
    for currency in TransactionItem.objects.values_list('currency', flat=True).distinct().order_by():
        TransactionEvent.objects.filter(currency=None, transaction__currency=currency).update(currency=currency)

class Migration(migrations.Migration):
    dependencies = [('payment', '0043_drop_from_state_renamed_fields')]
    operations = [migrations.RunPython(set_default_currency_for_transaction_event, migrations.RunPython.noop)]