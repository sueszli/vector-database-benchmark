import importlib.util
import logging
import re
import time
from collections import defaultdict
from graphlib import TopologicalSorter
from inspect import getsource
from pathlib import Path
from types import ModuleType
from typing import Any
import click
from flask import current_app
from flask_appbuilder import Model
from flask_migrate import downgrade, upgrade
from progress.bar import ChargingBar
from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.automap import automap_base
from superset import db
from superset.utils.mock_data import add_sample_rows
logger = logging.getLogger(__name__)

def import_migration_script(filepath: Path) -> ModuleType:
    if False:
        print('Hello World!')
    '\n    Import migration script as if it were a module.\n    '
    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
    if spec:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise Exception(f'No module spec found in location: `{str(filepath)}`')

def extract_modified_tables(module: ModuleType) -> set[str]:
    if False:
        return 10
    '\n    Extract the tables being modified by a migration script.\n\n    This function uses a simple approach of looking at the source code of\n    the migration script looking for patterns. It could be improved by\n    actually traversing the AST.\n    '
    tables: set[str] = set()
    for function in {'upgrade', 'downgrade'}:
        source = getsource(getattr(module, function))
        tables.update(re.findall('alter_table\\(\\s*"(\\w+?)"\\s*\\)', source, re.DOTALL))
        tables.update(re.findall('add_column\\(\\s*"(\\w+?)"\\s*,', source, re.DOTALL))
        tables.update(re.findall('drop_column\\(\\s*"(\\w+?)"\\s*,', source, re.DOTALL))
    return tables

def find_models(module: ModuleType) -> list[type[Model]]:
    if False:
        print('Hello World!')
    '\n    Find all models in a migration script.\n    '
    models: list[type[Model]] = []
    tables = extract_modified_tables(module)
    queue = list(module.__dict__.values())
    while queue:
        obj = queue.pop()
        if hasattr(obj, '__tablename__'):
            tables.add(obj.__tablename__)
        elif isinstance(obj, list):
            queue.extend(obj)
        elif isinstance(obj, dict):
            queue.extend(obj.values())
    sqlalchemy_uri = current_app.config['SQLALCHEMY_DATABASE_URI']
    engine = create_engine(sqlalchemy_uri)
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    seen = set()
    while tables:
        table = tables.pop()
        seen.add(table)
        try:
            model = getattr(Base.classes, table)
        except AttributeError:
            continue
        model.__tablename__ = table
        models.append(model)
        inspector = inspect(model)
        for column in inspector.columns.values():
            for foreign_key in column.foreign_keys:
                table = foreign_key.column.table.name
                if table not in seen:
                    tables.add(table)
    sorter: TopologicalSorter[Any] = TopologicalSorter()
    for model in models:
        inspector = inspect(model)
        dependent_tables: list[str] = []
        for column in inspector.columns.values():
            for foreign_key in column.foreign_keys:
                if foreign_key.column.table.name != model.__tablename__:
                    dependent_tables.append(foreign_key.column.table.name)
        sorter.add(model.__tablename__, *dependent_tables)
    order = list(sorter.static_order())
    models.sort(key=lambda model: order.index(model.__tablename__))
    return models

@click.command()
@click.argument('filepath')
@click.option('--limit', default=1000, help='Maximum number of entities.')
@click.option('--force', is_flag=True, help='Do not prompt for confirmation.')
@click.option('--no-auto-cleanup', is_flag=True, help='Do not remove created models.')
def main(filepath: str, limit: int=1000, force: bool=False, no_auto_cleanup: bool=False) -> None:
    if False:
        print('Hello World!')
    auto_cleanup = not no_auto_cleanup
    session = db.session()
    print(f'Importing migration script: {filepath}')
    module = import_migration_script(Path(filepath))
    revision: str = getattr(module, 'revision', '')
    down_revision: str = getattr(module, 'down_revision', '')
    if not revision or not down_revision:
        raise Exception("Not a valid migration script, couldn't find down_revision/revision")
    print(f'Migration goes from {down_revision} to {revision}')
    current_revision = db.engine.execute('SELECT version_num FROM alembic_version').scalar()
    print(f'Current version of the DB is {current_revision}')
    if current_revision != down_revision:
        if not force:
            click.confirm(f'\nRunning benchmark will downgrade the Superset DB to {down_revision} and upgrade to {revision} again. There may be data loss in downgrades. Continue?', abort=True)
        downgrade(revision=down_revision)
    print('\nIdentifying models used in the migration:')
    models = find_models(module)
    model_rows: dict[type[Model], int] = {}
    for model in models:
        rows = session.query(model).count()
        print(f'- {model.__name__} ({rows} rows in table {model.__tablename__})')
        model_rows[model] = rows
    session.close()
    print('Benchmarking migration')
    results: dict[str, float] = {}
    start = time.time()
    upgrade(revision=revision)
    duration = time.time() - start
    results['Current'] = duration
    print(f'Migration on current DB took: {duration:.2f} seconds')
    min_entities = 10
    new_models: dict[type[Model], list[Model]] = defaultdict(list)
    while min_entities <= limit:
        downgrade(revision=down_revision)
        print(f'Running with at least {min_entities} entities of each model')
        for model in models:
            missing = min_entities - model_rows[model]
            if missing > 0:
                entities: list[Model] = []
                print(f'- Adding {missing} entities to the {model.__name__} model')
                bar = ChargingBar('Processing', max=missing)
                try:
                    for entity in add_sample_rows(session, model, missing):
                        entities.append(entity)
                        bar.next()
                except Exception:
                    session.rollback()
                    raise
                bar.finish()
                model_rows[model] = min_entities
                session.add_all(entities)
                session.commit()
                if auto_cleanup:
                    new_models[model].extend(entities)
        start = time.time()
        upgrade(revision=revision)
        duration = time.time() - start
        print(f'Migration for {min_entities}+ entities took: {duration:.2f} seconds')
        results[f'{min_entities}+'] = duration
        min_entities *= 10
    print('\nResults:\n')
    for (label, duration) in results.items():
        print(f'{label}: {duration:.2f} s')
    if auto_cleanup:
        print('Cleaning up DB')
        for (model, entities) in list(new_models.items())[::-1]:
            session.query(model).filter(model.id.in_((entity.id for entity in entities))).delete(synchronize_session=False)
        session.commit()
    if current_revision != revision and (not force):
        click.confirm(f'\nRevert DB to {revision}?', abort=True)
        upgrade(revision=revision)
        print('Reverted')
if __name__ == '__main__':
    from superset.app import create_app
    app = create_app()
    with app.app_context():
        main()