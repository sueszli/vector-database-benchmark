"""add-k8s-yaml-to-rendered-templates

Revision ID: 45ba3f1493b9
Revises: 364159666cbd
Create Date: 2020-10-23 23:01:52.471442

"""
from __future__ import annotations
import sqlalchemy_jsonfield
from alembic import op
from sqlalchemy import Column
from airflow.settings import json
revision = '45ba3f1493b9'
down_revision = '364159666cbd'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'
__tablename__ = 'rendered_task_instance_fields'
k8s_pod_yaml = Column('k8s_pod_yaml', sqlalchemy_jsonfield.JSONField(json=json), nullable=True)

def upgrade():
    if False:
        i = 10
        return i + 15
    'Apply add-k8s-yaml-to-rendered-templates'
    with op.batch_alter_table(__tablename__, schema=None) as batch_op:
        batch_op.add_column(k8s_pod_yaml)

def downgrade():
    if False:
        i = 10
        return i + 15
    'Unapply add-k8s-yaml-to-rendered-templates'
    with op.batch_alter_table(__tablename__, schema=None) as batch_op:
        batch_op.drop_column('k8s_pod_yaml')