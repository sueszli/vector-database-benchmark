import pandas as pd
from sqlalchemy import DateTime, inspect
import superset.utils.database as database_utils
from superset import db
from .helpers import get_example_url, get_table_connector_registry

def load_flights(only_metadata: bool=False, force: bool=False) -> None:
    if False:
        while True:
            i = 10
    'Loading random time series data from a zip file in the repo'
    tbl_name = 'flights'
    database = database_utils.get_example_database()
    with database.get_sqla_engine_with_context() as engine:
        schema = inspect(engine).default_schema_name
        table_exists = database.has_table_by_name(tbl_name)
        if not only_metadata and (not table_exists or force):
            flight_data_url = get_example_url('flight_data.csv.gz')
            pdf = pd.read_csv(flight_data_url, encoding='latin-1', compression='gzip')
            airports_url = get_example_url('airports.csv.gz')
            airports = pd.read_csv(airports_url, encoding='latin-1', compression='gzip')
            airports = airports.set_index('IATA_CODE')
            pdf['ds'] = pdf.YEAR.map(str) + '-0' + pdf.MONTH.map(str) + '-0' + pdf.DAY.map(str)
            pdf.ds = pd.to_datetime(pdf.ds)
            pdf.drop(columns=['DAY', 'MONTH', 'YEAR'])
            pdf = pdf.join(airports, on='ORIGIN_AIRPORT', rsuffix='_ORIG')
            pdf = pdf.join(airports, on='DESTINATION_AIRPORT', rsuffix='_DEST')
            pdf.to_sql(tbl_name, engine, schema=schema, if_exists='replace', chunksize=500, dtype={'ds': DateTime}, index=False)
    table = get_table_connector_registry()
    tbl = db.session.query(table).filter_by(table_name=tbl_name).first()
    if not tbl:
        tbl = table(table_name=tbl_name, schema=schema)
    tbl.description = 'Random set of flights in the US'
    tbl.database = database
    tbl.filter_select_enabled = True
    db.session.merge(tbl)
    db.session.commit()
    tbl.fetch_metadata()
    print('Done loading table!')