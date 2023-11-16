import InvenTree.fields
from django.db import migrations
import djmoney.models.fields
from django.db.migrations.recorder import MigrationRecorder

def show_migrations(apps, schema_editor):
    if False:
        while True:
            i = 10
    'Show the latest migrations from each app'
    for app in apps.get_app_configs():
        label = app.label
        migrations = MigrationRecorder.Migration.objects.filter(app=app).order_by('-applied')[:5]
        print(f'{label} migrations:')
        for m in migrations:
            print(f' - {m.name}')

class Migration(migrations.Migration):
    dependencies = [('stock', '0064_auto_20210621_1724')]
    operations = []
    xoperations = [migrations.RunPython(code=show_migrations, reverse_code=migrations.RunPython.noop), migrations.AlterField(model_name='stockitem', name='purchase_price', field=InvenTree.fields.InvenTreeModelMoneyField(blank=True, currency_choices=[], decimal_places=4, default_currency='', help_text='Single unit purchase price at time of purchase', max_digits=19, null=True, verbose_name='Purchase Price')), migrations.AlterField(model_name='stockitem', name='purchase_price_currency', field=djmoney.models.fields.CurrencyField(choices=[], default='', editable=False, max_length=3))]