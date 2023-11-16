from __future__ import annotations
import argparse
import datetime
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
import requests
import sqlalchemy as sa
import toolz
if TYPE_CHECKING:
    from collections.abc import Mapping
SCHEMAS = {'countries': [('iso_alpha2', sa.TEXT), ('iso_alpha3', sa.TEXT), ('iso_numeric', sa.INT), ('fips', sa.TEXT), ('name', sa.TEXT), ('capital', sa.TEXT), ('area_km2', sa.REAL), ('population', sa.INT), ('continent', sa.TEXT)], 'gdp': [('country_code', sa.TEXT), ('year', sa.INT), ('value', sa.REAL)], 'independence': [('country_code', sa.TEXT), ('independence_date', sa.DATE), ('independence_from', sa.TEXT)]}
POST_PARSE_FUNCTIONS = {'independence': lambda row: toolz.assoc(row, 'independence_date', datetime.datetime.strptime(row['independence_date'], '%Y-%m-%d').date())}

def make_geography_db(data: Mapping[str, Any], con: sa.engine.Engine) -> None:
    if False:
        return 10
    metadata = sa.MetaData(bind=con)
    with con.begin() as bind:
        for (table_name, schema) in SCHEMAS.items():
            table = sa.Table(table_name, metadata, *(sa.Column(col_name, col_type) for (col_name, col_type) in schema))
            table_columns = table.c.keys()
            post_parse = POST_PARSE_FUNCTIONS.get(table_name, toolz.identity)
            table.drop(bind=bind, checkfirst=True)
            table.create(bind=bind)
            bind.execute(table.insert().values(), [post_parse(dict(zip(table_columns, row))) for row in data[table_name]])

def main() -> None:
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Create the geography SQLite database for the Ibis tutorial')
    parser.add_argument('-d', '--output-directory', default=tempfile.gettempdir(), type=str, help='The directory to which the database will be output')
    parser.add_argument('-u', '--input-data-url', default='https://storage.googleapis.com/ibis-tutorial-data/geography.json', type=str, help='The URL containing the data with which to populate the database')
    args = parser.parse_args()
    response = requests.get(args.input_data_url)
    response.raise_for_status()
    input_data = response.json()
    db_path = Path(args.output_directory).joinpath('geography.db')
    con = sa.create_engine(f'sqlite:///{db_path}')
    make_geography_db(input_data, con)
    print(db_path)
if __name__ == '__main__':
    main()