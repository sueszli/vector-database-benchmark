from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def clear_preregistrationuser_invited_as_admin(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        i = 10
        return i + 15
    'This migration fixes any PreregistrationUser objects that might\n    have been already corrupted to have the administrator role by the\n    buggy original version of migration\n    0198_preregistrationuser_invited_as.\n\n    Since invitations that create new users as administrators are\n    rare, it is cleaner to just remove the role from all\n    PreregistrationUser objects than to filter for just those older\n    invitation objects that could have been corrupted by the original\n    migration, which would have been possible using the\n    django_migrations table to check the date when the buggy migration\n    was run.\n    '
    INVITED_AS_MEMBER = 1
    INVITED_AS_REALM_ADMIN = 2
    PreregistrationUser = apps.get_model('zerver', 'PreregistrationUser')
    PreregistrationUser.objects.filter(invited_as=INVITED_AS_REALM_ADMIN).update(invited_as=INVITED_AS_MEMBER)

class Migration(migrations.Migration):
    dependencies = [('zerver', '0260_missed_message_addresses_from_redis_to_db')]
    operations = [migrations.RunPython(clear_preregistrationuser_invited_as_admin, reverse_code=migrations.RunPython.noop)]