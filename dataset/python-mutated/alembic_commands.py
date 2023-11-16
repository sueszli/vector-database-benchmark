from functools import wraps
from pathlib import Path
from threading import Lock
import prefect
ALEMBIC_LOCK = Lock()

def with_alembic_lock(fn):
    if False:
        for i in range(10):
            print('nop')
    "\n    Decorator that prevents alembic commands from running concurrently.\n    This is necessary because alembic uses a global configuration object\n    that is not thread-safe.\n\n    This issue occurred in https://github.com/PrefectHQ/prefect-dask/pull/50, where\n    dask threads were simultaneously performing alembic upgrades, and causing\n    cryptic `KeyError: 'config'` when `del globals_[attr_name]`.\n    "

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if False:
            return 10
        with ALEMBIC_LOCK:
            return fn(*args, **kwargs)
    return wrapper

def alembic_config():
    if False:
        print('Hello World!')
    from alembic.config import Config
    alembic_dir = Path(prefect.server.database.__file__).parent
    if not alembic_dir.joinpath('alembic.ini').exists():
        raise ValueError(f"Couldn't find alembic.ini at {alembic_dir}/alembic.ini")
    alembic_cfg = Config(alembic_dir / 'alembic.ini')
    return alembic_cfg

@with_alembic_lock
def alembic_upgrade(revision: str='head', dry_run: bool=False):
    if False:
        i = 10
        return i + 15
    "\n    Run alembic upgrades on Prefect REST API database\n\n    Args:\n        revision: The revision passed to `alembic downgrade`. Defaults to 'head', upgrading all revisions.\n        dry_run: Show what migrations would be made without applying them. Will emit sql statements to stdout.\n    "
    import alembic.command
    alembic.command.upgrade(alembic_config(), revision, sql=dry_run)

@with_alembic_lock
def alembic_downgrade(revision: str='base', dry_run: bool=False):
    if False:
        i = 10
        return i + 15
    "\n    Run alembic downgrades on Prefect REST API database\n\n    Args:\n        revision: The revision passed to `alembic downgrade`. Defaults to 'base', downgrading all revisions.\n        dry_run: Show what migrations would be made without applying them. Will emit sql statements to stdout.\n    "
    import alembic.command
    alembic.command.downgrade(alembic_config(), revision, sql=dry_run)

@with_alembic_lock
def alembic_revision(message: str=None, autogenerate: bool=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Create a new revision file for the database.\n\n    Args:\n        message: string message to apply to the revision.\n        autogenerate: whether or not to autogenerate the script from the database.\n    '
    import alembic.command
    alembic.command.revision(alembic_config(), message=message, autogenerate=autogenerate, **kwargs)

@with_alembic_lock
def alembic_stamp(revision):
    if False:
        print('Hello World!')
    "\n    Stamp the revision table with the given revision; don't run any migrations\n\n    Args:\n        revision: The revision passed to `alembic stamp`.\n    "
    import alembic.command
    alembic.command.stamp(alembic_config(), revision=revision)