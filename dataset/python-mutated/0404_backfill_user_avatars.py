from django.db import migrations
from sentry.api.utils import generate_region_url
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_user_avatar(apps, schema_editor):
    if False:
        while True:
            i = 10
    User = apps.get_model('sentry', 'User')
    UserAvatar = apps.get_model('sentry', 'UserAvatar')
    for user in RangeQuerySetWrapperWithProgressBar(User.objects.all()):
        avatar = UserAvatar.objects.filter(user_id=user.id).first()
        if avatar is None:
            continue
        user.avatar_type = avatar.avatar_type
        if avatar.avatar_type == 1:
            user.avatar_url = f'{generate_region_url()}/avatar/{avatar.ident}/'
        user.save(update_fields=['avatar_url', 'avatar_type'])

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0403_backfill_actors')]
    operations = [migrations.RunPython(backfill_user_avatar, reverse_code=migrations.RunPython.noop, hints={'tables': ['auth_user', 'sentry_useravatar']})]