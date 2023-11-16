"""Find unencrypted passwords in the database and delete them

Revision ID: 908b0085d28d
Revises: 1583a48cb978
Create Date: 2017-03-17 13:16:19.539970
Author: Mike Grima <mgrima@netflix.com>

"""
import re
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from alembic import op
revision = '908b0085d28d'
down_revision = '1583a48cb978'
Session = sessionmaker()
Base = declarative_base()

class User(Base):
    __tablename__ = 'user'
    id = sa.Column(sa.Integer, primary_key=True)
    email = sa.Column(sa.String(255), unique=True)
    password = sa.Column(sa.String(255))
    active = sa.Column(sa.Boolean())
    confirmed_at = sa.Column(sa.DateTime())
    daily_audit_email = sa.Column(sa.Boolean())
    change_reports = sa.Column(sa.String(32))
    last_login_at = sa.Column(sa.DateTime())
    current_login_at = sa.Column(sa.DateTime())
    login_count = sa.Column(sa.Integer)
    last_login_ip = sa.Column(sa.String(45))
    current_login_ip = sa.Column(sa.String(45))
    role = sa.Column(sa.String(30))

    def __str__(self):
        if False:
            return 10
        return '<User id=%s email=%s>' % (self.id, self.email)

def upgrade():
    if False:
        print('Hello World!')
    print('[@] Checking for users that have non-bcrypted/plaintext passwords in the database.')
    bind = op.get_bind()
    session = Session(bind=bind)
    users = session.query(User).all()
    for user in users:
        if user.password:
            if not re.match('^\\$2[ayb]\\$.{56}$', user.password):
                print('[!] User: {} has a plaintext password! Deleting the password!'.format(user.email))
                user.password = ''
                session.add(user)
                session.commit()
                print("[-] Deleted plaintext password from user: {}'s account".format(user.email))
    print('[@] Completed plaintext password check.')

def downgrade():
    if False:
        print('Hello World!')
    pass