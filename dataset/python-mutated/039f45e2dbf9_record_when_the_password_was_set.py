"""
Record when the password was set

Revision ID: 039f45e2dbf9
Revises: a65114e48d6f
Create Date: 2016-06-15 13:10:02.361621
"""
import sqlalchemy as sa
from alembic import op
revision = '039f45e2dbf9'
down_revision = 'a65114e48d6f'

def upgrade():
    if False:
        return 10
    op.add_column('accounts_user', sa.Column('password_date', sa.DateTime(), nullable=True))
    op.alter_column('accounts_user', 'password_date', server_default=sa.text('now()'))
    op.execute(' CREATE FUNCTION update_password_date()\n            RETURNS TRIGGER AS $$\n                BEGIN\n                    NEW.password_date = now();\n                    RETURN NEW;\n                END;\n            $$ LANGUAGE plpgsql;\n        ')
    op.execute(' CREATE TRIGGER update_user_password_date\n            BEFORE UPDATE OF password ON accounts_user\n            FOR EACH ROW\n            WHEN (OLD.password IS DISTINCT FROM NEW.password)\n            EXECUTE PROCEDURE update_password_date()\n        ')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    raise RuntimeError('Order No. 227 - Ни шагу назад!')