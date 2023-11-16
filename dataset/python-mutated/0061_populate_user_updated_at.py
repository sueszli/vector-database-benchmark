from django.db import migrations
from django.db.models import F

def populate_user_updated_at_datetimes(apps, _schema_editor):
    if False:
        while True:
            i = 10
    User = apps.get_model('account', 'User')
    User.objects.filter(updated_at__isnull=True).update(updated_at=F('date_joined'))

class Migration(migrations.Migration):
    dependencies = [('account', '0060_user_updated_at')]
    operations = [migrations.RunPython(populate_user_updated_at_datetimes, migrations.RunPython.noop)]