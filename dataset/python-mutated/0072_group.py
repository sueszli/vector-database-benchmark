from django.db import migrations, models
import saleor.account.models

def rename_group_tables(apps, schema_editor):
    if False:
        print('Hello World!')
    Group = apps.get_model('auth', 'Group')
    schema_editor.alter_db_table(Group, 'auth_group', 'account_group')
    PermissionGroup = Group.permissions.through
    schema_editor.alter_db_table(PermissionGroup, 'auth_group_permissions', 'account_group_permissions')
RENAME_CONSTRAINTS_AND_INDEX = '\nALTER TABLE account_group RENAME CONSTRAINT auth_group_pkey\n    TO account_group_pkey;\n\nALTER TABLE account_group RENAME CONSTRAINT auth_group_name_key\n    TO account_group_name_key;\n\nALTER INDEX IF EXISTS auth_group_name_a6ea08ec_like\n    RENAME TO account_group_name_034e9f3f_like;\n'

def rename_group_tables_reverse(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Group = apps.get_model('auth', 'Group')
    schema_editor.alter_db_table(Group, 'account_group', 'auth_group')
    PermissionGroup = Group.permissions.through
    schema_editor.alter_db_table(PermissionGroup, 'account_group_permissions', 'auth_group_permissions')
RENAME_CONSTRAINTS_AND_INDEX_REVERSE = '\nALTER TABLE account_group RENAME CONSTRAINT account_group_pkey\n    TO auth_group_pkey;\n\nALTER TABLE account_group RENAME CONSTRAINT account_group_name_key\n    TO auth_group_name_key;\n\nALTER INDEX IF EXISTS account_group_name_034e9f3f_like\n    RENAME TO auth_group_name_a6ea08ec_like;\n'
DROP_OLD_CONSTRAINTS_REVERSE_FROM_0073 = '\nALTER TABLE auth_group_permissions\n    ADD CONSTRAINT auth_group_permissions_group_id_permission_id_0cd325b0_uniq\n    UNIQUE (group_id, permission_id);\n\nALTER TABLE auth_group_permissions\n    ADD CONSTRAINT auth_group_permissions_group_id_b120cbf9_fk_auth_group_id\n    FOREIGN KEY (group_id) REFERENCES auth_group (id) DEFERRABLE INITIALLY DEFERRED;\n\nALTER TABLE auth_group_permissions\n    ADD CONSTRAINT auth_group_permissio_permission_id_84c5c92e_fk_auth_perm\n    FOREIGN KEY (permission_id) REFERENCES auth_permission (id)\n    DEFERRABLE INITIALLY DEFERRED;\n\nALTER TABLE account_user_groups\n    ADD CONSTRAINT userprofile_user_groups_group_id_c7eec74e_fk_auth_group_id\n    FOREIGN KEY (group_id) REFERENCES auth_group (id) DEFERRABLE INITIALLY DEFERRED;\n\nALTER TABLE account_user_user_permissions\n    ADD CONSTRAINT userprofile_user_use_permission_id_1caa8a71_fk_auth_perm\n    FOREIGN KEY (permission_id) REFERENCES auth_permission (id)\n    DEFERRABLE INITIALLY DEFERRED;\n'
DROP_OLD_CONSTRAINTS_REVERSE_FROM_APP_0018 = '\nALTER TABLE app_app_permissions\n    ADD CONSTRAINT account_serviceaccou_permission_id_449791f0_fk_auth_perm\n    FOREIGN KEY (permission_id) REFERENCES auth_permission (id)\n    DEFERRABLE INITIALLY DEFERRED;\n\nALTER TABLE app_appextension_permissions\n    ADD CONSTRAINT app_appextension_per_permission_id_cb6c3ce0_fk_auth_perm\n    FOREIGN KEY (permission_id) REFERENCES auth_permission (id)\n    DEFERRABLE INITIALLY DEFERRED;\n\nALTER TABLE app_appinstallation_permissions\n    ADD CONSTRAINT app_appinstallation__permission_id_4ee9f6c8_fk_auth_perm\n    FOREIGN KEY (permission_id) REFERENCES auth_permission (id)\n    DEFERRABLE INITIALLY DEFERRED;\n'

class Migration(migrations.Migration):
    dependencies = [('account', '0071_user_external_reference'), ('app', '0017_app_audience'), ('auth', '0012_alter_user_first_name_max_length')]
    operations = [migrations.SeparateDatabaseAndState(database_operations=[migrations.RunSQL(migrations.RunSQL.noop, reverse_sql=DROP_OLD_CONSTRAINTS_REVERSE_FROM_0073), migrations.RunSQL(migrations.RunSQL.noop, reverse_sql=DROP_OLD_CONSTRAINTS_REVERSE_FROM_APP_0018), migrations.RunPython(rename_group_tables, rename_group_tables_reverse), migrations.RunSQL(RENAME_CONSTRAINTS_AND_INDEX, reverse_sql=RENAME_CONSTRAINTS_AND_INDEX_REVERSE)], state_operations=[migrations.CreateModel(name='Group', fields=[('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), ('name', models.CharField(max_length=150, unique=True, verbose_name='name')), ('permissions', models.ManyToManyField(blank=True, to='auth.Permission', verbose_name='permissions'))], options={'verbose_name': 'group', 'verbose_name_plural': 'groups'}, managers=[('objects', saleor.account.models.GroupManager())])])]