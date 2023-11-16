import json
import pandas as pd
import polyline
from sqlalchemy import inspect, String, Text
from superset import db
from ..utils.database import get_example_database
from .helpers import get_example_url, get_table_connector_registry

def load_bart_lines(only_metadata: bool=False, force: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    tbl_name = 'bart_lines'
    database = get_example_database()
    with database.get_sqla_engine_with_context() as engine:
        schema = inspect(engine).default_schema_name
        table_exists = database.has_table_by_name(tbl_name)
        if not only_metadata and (not table_exists or force):
            url = get_example_url('bart-lines.json.gz')
            df = pd.read_json(url, encoding='latin-1', compression='gzip')
            df['path_json'] = df.path.map(json.dumps)
            df['polyline'] = df.path.map(polyline.encode)
            del df['path']
            df.to_sql(tbl_name, engine, schema=schema, if_exists='replace', chunksize=500, dtype={'color': String(255), 'name': String(255), 'polyline': Text, 'path_json': Text}, index=False)
    print(f'Creating table {tbl_name} reference')
    table = get_table_connector_registry()
    tbl = db.session.query(table).filter_by(table_name=tbl_name).first()
    if not tbl:
        tbl = table(table_name=tbl_name, schema=schema)
    tbl.description = 'BART lines'
    tbl.database = database
    tbl.filter_select_enabled = True
    db.session.merge(tbl)
    db.session.commit()
    tbl.fetch_metadata()