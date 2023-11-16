import json
import pandas as pd
from sqlalchemy import BigInteger, Float, inspect, Text
import superset.utils.database as database_utils
from superset import db
from .helpers import get_example_url, get_table_connector_registry

def load_sf_population_polygons(only_metadata: bool=False, force: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    tbl_name = 'sf_population_polygons'
    database = database_utils.get_example_database()
    with database.get_sqla_engine_with_context() as engine:
        schema = inspect(engine).default_schema_name
        table_exists = database.has_table_by_name(tbl_name)
        if not only_metadata and (not table_exists or force):
            url = get_example_url('sf_population.json.gz')
            df = pd.read_json(url, compression='gzip')
            df['contour'] = df.contour.map(json.dumps)
            df.to_sql(tbl_name, engine, schema=schema, if_exists='replace', chunksize=500, dtype={'zipcode': BigInteger, 'population': BigInteger, 'contour': Text, 'area': Float}, index=False)
    print(f'Creating table {tbl_name} reference')
    table = get_table_connector_registry()
    tbl = db.session.query(table).filter_by(table_name=tbl_name).first()
    if not tbl:
        tbl = table(table_name=tbl_name, schema=schema)
    tbl.description = 'Population density of San Francisco'
    tbl.database = database
    tbl.filter_select_enabled = True
    db.session.merge(tbl)
    db.session.commit()
    tbl.fetch_metadata()