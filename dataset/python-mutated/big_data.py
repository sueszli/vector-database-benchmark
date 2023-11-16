import random
import string
import sqlalchemy.sql.sqltypes
from superset.utils.mock_data import add_data, ColumnInfo
COLUMN_TYPES = [sqlalchemy.sql.sqltypes.INTEGER(), sqlalchemy.sql.sqltypes.VARCHAR(length=255), sqlalchemy.sql.sqltypes.TEXT(), sqlalchemy.sql.sqltypes.BOOLEAN(), sqlalchemy.sql.sqltypes.FLOAT(), sqlalchemy.sql.sqltypes.DATE(), sqlalchemy.sql.sqltypes.TIME(), sqlalchemy.sql.sqltypes.TIMESTAMP()]

def load_big_data() -> None:
    if False:
        i = 10
        return i + 15
    print('Creating table `wide_table` with 100 columns')
    columns: list[ColumnInfo] = []
    for i in range(100):
        column: ColumnInfo = {'name': f'col{i}', 'type': COLUMN_TYPES[i % len(COLUMN_TYPES)], 'nullable': False, 'default': None, 'autoincrement': 'auto', 'primary_key': 1 if i == 0 else 0}
        columns.append(column)
    add_data(columns=columns, num_rows=1000, table_name='wide_table')
    print('Creating 1000 small tables')
    columns = [{'name': 'id', 'type': sqlalchemy.sql.sqltypes.INTEGER(), 'nullable': False, 'default': None, 'autoincrement': 'auto', 'primary_key': 1}, {'name': 'value', 'type': sqlalchemy.sql.sqltypes.VARCHAR(length=255), 'nullable': False, 'default': None, 'autoincrement': 'auto', 'primary_key': 0}]
    for i in range(1000):
        add_data(columns=columns, num_rows=10, table_name=f'small_table_{i}')
    print('Creating table with long name')
    name = ''.join(random.choices(string.ascii_letters + string.digits, k=60))
    add_data(columns=columns, num_rows=10, table_name=name)