"""css templates

Revision ID: d827694c7555
Revises: 43df8de3a5f4
Create Date: 2016-02-03 17:41:10.944019

"""
revision = 'd827694c7555'
down_revision = '43df8de3a5f4'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    op.create_table('css_templates', sa.Column('created_on', sa.DateTime(), nullable=False), sa.Column('changed_on', sa.DateTime(), nullable=False), sa.Column('id', sa.Integer(), nullable=False), sa.Column('template_name', sa.String(length=250), nullable=True), sa.Column('css', sa.Text(), nullable=True), sa.Column('changed_by_fk', sa.Integer(), nullable=True), sa.Column('created_by_fk', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['changed_by_fk'], ['ab_user.id']), sa.ForeignKeyConstraint(['created_by_fk'], ['ab_user.id']), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('css_templates')