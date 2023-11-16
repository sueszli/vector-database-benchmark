"""empty message

Revision ID: 11672b13f92c
Revises: 
Create Date: 2021-05-18 15:06:30.858112

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '11672b13f92c'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('commits',
    sa.Column('uuid', sa.Text(), nullable=False),
    sa.Column('hash', sa.Text(), nullable=True),
    sa.Column('experiment_name', sa.Text(), nullable=True),
    sa.Column('session_started_at', sa.Integer(), nullable=True),
    sa.Column('session_closed_at', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('is_archived', sa.Boolean(), nullable=True),
    sa.PrimaryKeyConstraint('uuid'),
    sa.UniqueConstraint('experiment_name', 'hash')
    )
    op.create_table('tags',
    sa.Column('uuid', sa.Text(), nullable=False),
    sa.Column('name', sa.Text(), nullable=True),
    sa.Column('color', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('is_archived', sa.Boolean(), nullable=True),
    sa.Column('is_hidden', sa.Boolean(), nullable=True),
    sa.PrimaryKeyConstraint('uuid')
    )
    op.create_table('tf_summary_logs',
    sa.Column('uuid', sa.Text(), nullable=False),
    sa.Column('log_path', sa.Text(), nullable=True),
    sa.Column('params', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('is_archived', sa.Boolean(), nullable=True),
    sa.PrimaryKeyConstraint('uuid')
    )
    op.create_table('commit_tag',
    sa.Column('commit_id', sa.Text(), nullable=True),
    sa.Column('tag_id', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['commit_id'], ['commits.uuid'], ),
    sa.ForeignKeyConstraint(['tag_id'], ['tags.uuid'], )
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('commit_tag')
    op.drop_table('tf_summary_logs')
    op.drop_table('tags')
    op.drop_table('commits')
    # ### end Alembic commands ###
