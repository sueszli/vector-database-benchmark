"""Please run "monkey fetch_aws_canonical_ids"

Revision ID: b8ccf5b8089b
Revises: 908b0085d28d
Create Date: 2017-03-23 11:00:43.792538
Author: Mike Grima <mgrima@netflix.com>, No-op'ed by Patrick

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
revision = 'b8ccf5b8089b'
down_revision = '908b0085d28d'

def upgrade():
    if False:
        print('Hello World!')
    pass

def downgrade():
    if False:
        while True:
            i = 10
    pass