from django.db import migrations, transaction
from django.db.models import QuerySet
from ..models import User
BATCH_SIZE = 5000

def queryset_in_batches(queryset):
    if False:
        i = 10
        return i + 15
    'Slice a queryset into batches.\n\n    Input queryset should be sorted be pk.\n    '
    start_pk = 0
    while True:
        qs = queryset.filter(pk__gt=start_pk)[:BATCH_SIZE]
        pks = list(qs.values_list('pk', flat=True))
        if not pks:
            break
        yield pks
        start_pk = pks[-1]

def set_user_is_confirmed_to_false(qs: QuerySet['User']):
    if False:
        for i in range(10):
            print('nop')
    with transaction.atomic():
        _users = list(qs.select_for_update(of=['self']))
        qs.update(is_confirmed=False)

def set_user_is_confirmed_task(apps, schema_editor):
    if False:
        return 10
    User = apps.get_model('account', 'User')
    SiteSettings = apps.get_model('site', 'SiteSettings')
    confirmation_enabled = SiteSettings.objects.first().enable_account_confirmation_by_email
    users = User.objects.order_by('pk').filter(is_confirmed=True)
    if confirmation_enabled:
        users = users.filter(is_active=False, last_login__isnull=True)
    for ids in queryset_in_batches(users):
        qs = User.objects.filter(pk__in=ids)
        set_user_is_confirmed_to_false(qs)

class Migration(migrations.Migration):
    dependencies = [('account', '0080_user_is_confirmed'), ('site', '0038_auto_20230510_1107')]
    operations = [migrations.RunPython(set_user_is_confirmed_task, migrations.RunPython.noop)]