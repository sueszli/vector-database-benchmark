import json
from pathlib import Path
from typing import Any
from rich import get_console
from sqlalchemy import create_engine
from sqlalchemy.orm import Mapped, Session, sessionmaker
from litestar.contrib.sqlalchemy.base import UUIDBase
from litestar.contrib.sqlalchemy.repository import SQLAlchemySyncRepository
from litestar.repository.filters import LimitOffset
here = Path(__file__).parent
console = get_console()

class USState(UUIDBase):
    __tablename__ = 'us_state_lookup'
    abbreviation: Mapped[str]
    name: Mapped[str]

class USStateRepository(SQLAlchemySyncRepository[USState]):
    """US State repository."""
    model_type = USState
engine = create_engine('duckdb:///:memory:', future=True)
session_factory: sessionmaker[Session] = sessionmaker(engine, expire_on_commit=False)

def open_fixture(fixtures_path: Path, fixture_name: str) -> Any:
    if False:
        print('Hello World!')
    'Loads JSON file with the specified fixture name\n\n    Args:\n        fixtures_path (Path): The path to look for fixtures\n        fixture_name (str): The fixture name to load.\n\n    Raises:\n        FileNotFoundError: Fixtures not found.\n\n    Returns:\n        Any: The parsed JSON data\n    '
    fixture = Path(fixtures_path / f'{fixture_name}.json')
    if fixture.exists():
        with fixture.open(mode='r', encoding='utf-8') as f:
            f_data = f.read()
        return json.loads(f_data)
    raise FileNotFoundError(f'Could not find the {fixture_name} fixture')

def run_script() -> None:
    if False:
        i = 10
        return i + 15
    'Load data from a fixture.'
    with engine.begin() as conn:
        USState.metadata.create_all(conn)
    with session_factory() as db_session:
        repo = USStateRepository(session=db_session)
        fixture = open_fixture(here, USStateRepository.model_type.__tablename__)
        objs = repo.add_many([USStateRepository.model_type(**raw_obj) for raw_obj in fixture])
        db_session.commit()
        console.print(f'Created {len(objs)} new objects.')
        (created_objs, total_objs) = repo.list_and_count(LimitOffset(limit=10, offset=0))
        console.print(f'Selected {len(created_objs)} records out of a total of {total_objs}.')
        deleted_objs = repo.delete_many([new_obj.id for new_obj in created_objs])
        console.print(f'Removed {len(deleted_objs)} records out of a total of {total_objs}.')
        remaining_count = repo.count()
        console.print(f'Found {remaining_count} remaining records after delete.')
if __name__ == '__main__':
    run_script()