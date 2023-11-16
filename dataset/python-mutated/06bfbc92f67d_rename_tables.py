"""
Rename tables

Revision ID: 06bfbc92f67d
Revises: eeb23d9b4d00
Create Date: 2018-11-06 04:36:58.531272
"""
from alembic import op
revision = '06bfbc92f67d'
down_revision = 'e612a92c1017'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.rename_table('packages', 'projects')
    op.execute('ALTER TABLE projects RENAME CONSTRAINT packages_pkey TO projects_pkey')
    op.execute('\n        ALTER TABLE projects\n            RENAME CONSTRAINT packages_valid_name\n            TO projects_valid_name\n        ')
    op.execute(" CREATE OR REPLACE FUNCTION maintain_project_last_serial()\n            RETURNS TRIGGER AS $$\n            DECLARE\n                targeted_name text;\n            BEGIN\n                IF TG_OP = 'INSERT' THEN\n                    targeted_name := NEW.name;\n                ELSEIF TG_OP = 'UPDATE' THEN\n                    targeted_name := NEW.name;\n                ELSIF TG_OP = 'DELETE' THEN\n                    targeted_name := OLD.name;\n                END IF;\n\n                UPDATE projects\n                SET last_serial = j.last_serial\n                FROM (\n                    SELECT max(id) as last_serial\n                    FROM journals\n                    WHERE journals.name = targeted_name\n                ) as j\n                WHERE projects.name = targeted_name;\n\n                RETURN NULL;\n            END;\n            $$ LANGUAGE plpgsql;\n        ")
    op.execute("UPDATE row_counts SET table_name = 'projects' WHERE table_name = 'packages'")
    op.rename_table('accounts_user', 'users')
    op.execute('ALTER TABLE users RENAME CONSTRAINT accounts_user_pkey TO users_pkey')
    op.execute('\n        ALTER TABLE users\n            RENAME CONSTRAINT accounts_user_username_key\n            TO users_username_key\n        ')
    op.execute('\n        ALTER TABLE users\n            RENAME CONSTRAINT accounts_user_valid_username\n            TO users_valid_username\n        ')
    op.execute('\n        ALTER TABLE users\n            RENAME CONSTRAINT packages_valid_name\n            TO users_valid_username_length\n        ')
    op.execute("UPDATE row_counts SET table_name = 'users' WHERE table_name = 'accounts_user'")
    op.rename_table('accounts_email', 'user_emails')
    op.execute('\n        ALTER TABLE user_emails\n            RENAME CONSTRAINT accounts_email_pkey\n            TO user_emails_pkey\n        ')
    op.execute('\n        ALTER TABLE user_emails\n            RENAME CONSTRAINT accounts_email_email_key\n            TO user_emails_email_key\n        ')
    op.execute('\n        ALTER TABLE user_emails\n            RENAME CONSTRAINT accounts_email_user_id_fkey\n            TO user_emails_user_id_fkey\n        ')
    op.execute('ALTER INDEX accounts_email_user_id RENAME TO user_emails_user_id')
    op.rename_table('warehouse_admin_flag', 'admin_flags')
    op.execute('\n        ALTER TABLE admin_flags\n            RENAME CONSTRAINT warehouse_admin_flag_pkey\n            TO admin_flags_pkey\n        ')
    op.rename_table('warehouse_admin_squat', 'admin_squats')
    op.execute('\n        ALTER TABLE admin_squats\n            RENAME CONSTRAINT warehouse_admin_squat_pkey\n            TO admin_squats_pkey\n        ')
    op.execute('\n        ALTER TABLE admin_squats\n            RENAME CONSTRAINT warehouse_admin_squat_squattee_id_fkey\n            TO admin_squats_squattee_id_fkey\n        ')
    op.execute('\n        ALTER TABLE admin_squats\n            RENAME CONSTRAINT warehouse_admin_squat_squatter_id_fkey\n            TO admin_squats_squatter_id_fkey\n        ')

def downgrade():
    if False:
        return 10
    raise RuntimeError('Order No. 227 - Ни шагу назад!')