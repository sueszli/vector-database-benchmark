from django.db import migrations
from django.db.models import Q

def set_order_settings(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    SiteSettings = apps.get_model('site', 'SiteSettings')
    Channel = apps.get_model('channel', 'Channel')
    site_settings = SiteSettings.objects.first()
    automatically_confirm_all_new_orders = True
    automatically_fulfill_non_shippable_gift_card = True
    if site_settings:
        automatically_confirm_all_new_orders = site_settings.automatically_confirm_all_new_orders
        automatically_fulfill_non_shippable_gift_card = site_settings.automatically_fulfill_non_shippable_gift_card
    Channel.objects.filter(Q(automatically_confirm_all_new_orders__isnull=True) | Q(automatically_fulfill_non_shippable_gift_card__isnull=True)).update(automatically_confirm_all_new_orders=automatically_confirm_all_new_orders, automatically_fulfill_non_shippable_gift_card=automatically_fulfill_non_shippable_gift_card)

class Migration(migrations.Migration):
    dependencies = [('channel', '0007_order_settings_per_channel')]
    operations = [migrations.RunPython(set_order_settings, migrations.RunPython.noop)]