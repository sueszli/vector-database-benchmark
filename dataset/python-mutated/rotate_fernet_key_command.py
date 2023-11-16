"""Rotate Fernet key command."""
from __future__ import annotations
from sqlalchemy import select
from airflow.models import Connection, Variable
from airflow.utils import cli as cli_utils
from airflow.utils.providers_configuration_loader import providers_configuration_loaded
from airflow.utils.session import create_session

@cli_utils.action_cli
@providers_configuration_loaded
def rotate_fernet_key(args):
    if False:
        i = 10
        return i + 15
    'Rotates all encrypted connection credentials and variables.'
    with create_session() as session:
        conns_query = select(Connection).where(Connection.is_encrypted | Connection.is_extra_encrypted)
        for conn in session.scalars(conns_query):
            conn.rotate_fernet_key()
        for var in session.scalars(select(Variable).where(Variable.is_encrypted)):
            var.rotate_fernet_key()