from django.db import migrations
from django_countries import countries

def get_countries_without_shipping_zone(ShippingZone):
    if False:
        i = 10
        return i + 15
    'Return countries that are not assigned to any shipping zone.'
    covered_countries = set()
    for zone in ShippingZone.objects.all():
        covered_countries.update({c.code for c in zone.countries})
    return (country[0] for country in countries if country[0] not in covered_countries)

def assign_countries_in_default_shipping_zone(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    ShippingZone = apps.get_model('shipping', 'ShippingZone')
    qs = ShippingZone.objects.filter(default=True)
    if qs.exists():
        default_zone = qs[0]
        if not default_zone.countries:
            default_zone.countries = get_countries_without_shipping_zone(ShippingZone)
            default_zone.save(update_fields=['countries'])

class Migration(migrations.Migration):
    dependencies = [('shipping', '0017_django_price_2')]
    operations = [migrations.RunPython(assign_countries_in_default_shipping_zone, migrations.RunPython.noop)]