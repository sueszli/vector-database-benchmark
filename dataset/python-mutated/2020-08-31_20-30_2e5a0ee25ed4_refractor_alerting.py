"""refractor_alerting

Revision ID: 2e5a0ee25ed4
Revises: f80a3b88324b
Create Date: 2020-08-31 20:30:30.781478

"""
revision = '2e5a0ee25ed4'
down_revision = 'f80a3b88324b'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

def upgrade():
    if False:
        print('Hello World!')
    op.create_table('alert_validators', sa.Column('created_on', sa.DateTime(), nullable=True), sa.Column('changed_on', sa.DateTime(), nullable=True), sa.Column('id', sa.Integer(), nullable=False), sa.Column('validator_type', sa.String(length=100), nullable=False), sa.Column('config', sa.Text(), nullable=True), sa.Column('created_by_fk', sa.Integer(), nullable=True), sa.Column('changed_by_fk', sa.Integer(), nullable=True), sa.Column('alert_id', sa.Integer(), nullable=False), sa.ForeignKeyConstraint(['alert_id'], ['alerts.id']), sa.ForeignKeyConstraint(['changed_by_fk'], ['ab_user.id']), sa.ForeignKeyConstraint(['created_by_fk'], ['ab_user.id']), sa.PrimaryKeyConstraint('id'))
    op.create_table('sql_observers', sa.Column('created_on', sa.DateTime(), nullable=True), sa.Column('changed_on', sa.DateTime(), nullable=True), sa.Column('id', sa.Integer(), nullable=False), sa.Column('sql', sa.Text(), nullable=False), sa.Column('created_by_fk', sa.Integer(), nullable=True), sa.Column('changed_by_fk', sa.Integer(), nullable=True), sa.Column('alert_id', sa.Integer(), nullable=False), sa.Column('database_id', sa.Integer(), nullable=False), sa.ForeignKeyConstraint(['alert_id'], ['alerts.id']), sa.ForeignKeyConstraint(['changed_by_fk'], ['ab_user.id']), sa.ForeignKeyConstraint(['created_by_fk'], ['ab_user.id']), sa.ForeignKeyConstraint(['database_id'], ['dbs.id']), sa.PrimaryKeyConstraint('id'))
    op.create_table('sql_observations', sa.Column('id', sa.Integer(), nullable=False), sa.Column('dttm', sa.DateTime(), nullable=True), sa.Column('observer_id', sa.Integer(), nullable=False), sa.Column('alert_id', sa.Integer(), nullable=True), sa.Column('value', sa.Float(), nullable=True), sa.Column('error_msg', sa.String(length=500), nullable=True), sa.ForeignKeyConstraint(['alert_id'], ['alerts.id']), sa.ForeignKeyConstraint(['observer_id'], ['sql_observers.id']), sa.PrimaryKeyConstraint('id'))
    op.create_index(op.f('ix_sql_observations_dttm'), 'sql_observations', ['dttm'], unique=False)
    with op.batch_alter_table('alerts') as batch_op:
        batch_op.add_column(sa.Column('changed_by_fk', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('changed_on', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('created_by_fk', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('created_on', sa.DateTime(), nullable=True))
        batch_op.alter_column('crontab', existing_type=mysql.VARCHAR(length=50), nullable=False)
        batch_op.create_foreign_key('alerts_ibfk_3', 'ab_user', ['changed_by_fk'], ['id'])
        batch_op.create_foreign_key('alerts_ibfk_4', 'ab_user', ['created_by_fk'], ['id'])
        batch_op.drop_column('sql')
        batch_op.drop_column('database_id')

def downgrade():
    if False:
        return 10
    with op.batch_alter_table('alerts') as batch_op:
        batch_op.add_column(sa.Column('database_id', mysql.INTEGER(), autoincrement=False, nullable=False))
        batch_op.add_column(sa.Column('sql', mysql.TEXT(), nullable=True))
        batch_op.drop_constraint('alerts_ibfk_3', type_='foreignkey')
        batch_op.drop_constraint('alerts_ibfk_4', type_='foreignkey')
        batch_op.alter_column('crontab', existing_type=mysql.VARCHAR(length=50), nullable=True)
        batch_op.drop_column('created_on')
        batch_op.drop_column('created_by_fk')
        batch_op.drop_column('changed_on')
        batch_op.drop_column('changed_by_fk')
    op.drop_index(op.f('ix_sql_observations_dttm'), table_name='sql_observations')
    op.drop_table('sql_observations')
    op.drop_table('sql_observers')
    op.drop_table('alert_validators')