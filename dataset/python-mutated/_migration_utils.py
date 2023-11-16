from awx.main.utils import set_current_apps

def set_current_apps_for_migrations(apps, schema_editor):
    if False:
        while True:
            i = 10
    "\n    This is necessary for migrations which do explicit saves on any model that\n    has an ImplicitRoleFIeld (which generally means anything that has\n    some RBAC bindings associated with it). This sets the current 'apps' that\n    the ImplicitRoleFIeld should be using when creating new roles.\n    "
    set_current_apps(apps)