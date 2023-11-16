import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
from lwe.backends.api.orm import Base
LWE_SCHEMA_MIGRATION_SQLALCHEMY_URL = os.environ.get('LWE_SCHEMA_MIGRATION_SQLALCHEMY_URL')
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    if False:
        while True:
            i = 10
    "Run migrations in 'offline' mode.\n\n    This configures the context with just a URL\n    and not an Engine, though an Engine is acceptable\n    here as well.  By skipping the Engine creation\n    we don't even need a DBAPI to be available.\n\n    Calls to context.execute() here emit the given string to the\n    script output.\n\n    "
    url = LWE_SCHEMA_MIGRATION_SQLALCHEMY_URL or config.get_main_option('sqlalchemy.url')
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, dialect_opts={'paramstyle': 'named'})
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    if False:
        for i in range(10):
            print('nop')
    "Run migrations in 'online' mode.\n\n    In this scenario we need to create an Engine\n    and associate a connection with the context.\n\n    "
    ini_config = config.get_section(config.config_ini_section, {})
    if LWE_SCHEMA_MIGRATION_SQLALCHEMY_URL:
        ini_config['sqlalchemy.url'] = LWE_SCHEMA_MIGRATION_SQLALCHEMY_URL
    connectable = engine_from_config(ini_config, prefix='sqlalchemy.', poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()