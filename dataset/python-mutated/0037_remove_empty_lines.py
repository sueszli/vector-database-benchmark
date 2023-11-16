from django.db import migrations

def remove_empty_lines(apps, schema_editor):
    if False:
        print('Hello World!')
    CheckoutLine = apps.get_model('checkout', 'CheckoutLine')
    CheckoutLine.objects.filter(quantity=0).delete()

class Migration(migrations.Migration):
    dependencies = [('checkout', '0036_alter_checkout_language_code')]
    operations = [migrations.RunPython(remove_empty_lines, migrations.RunPython.noop)]