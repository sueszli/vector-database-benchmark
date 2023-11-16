from django.db import migrations
from django.db.models import F
from django.utils import timezone

def populate_products_datetimes(apps, _schema_editor):
    if False:
        return 10
    Product = apps.get_model('product', 'Product')
    Product.objects.filter(updated_at__isnull=False).update(created=F('updated_at'))
    Product.objects.filter(created__isnull=True).update(created=timezone.now())
    Product.objects.filter(updated_at__isnull=True).update(updated_at=timezone.now())
    ProductVariant = apps.get_model('product', 'ProductVariant')
    ProductVariant.objects.update(created=timezone.now(), updated_at=timezone.now())

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('product', '0159_auto_20220209_1501')]
    operations = [migrations.RunPython(populate_products_datetimes, migrations.RunPython.noop)]