import sqlite3
from ansible.module_utils.basic import AnsibleModule
from lwe.core.config import Config
from lwe.core.logger import Logger
config = Config()
config.set('debug.log.enabled', True)
log = Logger('lwe_sqlite_query', config)
DOCUMENTATION = '\n---\nmodule: lwe_sqlite_query\nshort_description: Run a query against a SQLite database\ndescription:\n    - This module runs a query against a specified SQLite database and stores any returned data in a structured format.\noptions:\n    db:\n      description:\n          - The path to the SQLite database file.\n      type: str\n      required: true\n    query:\n      description:\n          - The SQL query to execute.\n      type: str\n      required: true\n    query_params:\n      description:\n          - Optional list of query params to pass to a parameterized query.\n      type: list\n      required: false\nauthor:\n    - Chad Phillips (@thehunmonkgroup)\n'
EXAMPLES = '\n  - name: Run a SELECT query against a SQLite database\n    lwe_sqlite_query:\n      db: "/path/to/your/database.db"\n      query: "SELECT * FROM your_table WHERE id = ?"\n      query_params:\n        - 1\n'
RETURN = '\n  data:\n      description: The data returned from the query.\n      type: list\n      returned: success\n  row_count:\n      description: The number of rows returned from the query.\n      type: int\n      returned: success\n'

def run_query(db, query, params=()):
    if False:
        i = 10
        return i + 15
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    if not query.lower().strip().startswith('select'):
        conn.commit()
    data = [dict(row) for row in cursor.fetchall()]
    row_count = len(data)
    conn.close()
    return (data, row_count)

def main():
    if False:
        for i in range(10):
            print('nop')
    result = dict(changed=False, response=dict())
    module = AnsibleModule(argument_spec=dict(db=dict(type='str', required=True), query=dict(type='str', required=True), query_params=dict(type='list', required=False, default=[])), supports_check_mode=True)
    db = module.params['db']
    query = module.params['query']
    query_params = module.params['query_params']
    if module.check_mode:
        module.exit_json(**result)
    try:
        log.debug(f'Running query on database: {db}: query: {query}, params: {query_params}')
        (data, row_count) = run_query(db, query, tuple(query_params))
        result['changed'] = True
        result['data'] = data
        result['row_count'] = row_count
        module.exit_json(**result)
    except Exception as e:
        result['failed'] = True
        message = f'Failed to run query: {query}, error: {e}'
        log.error(message)
        module.fail_json(msg=message, **result)
if __name__ == '__main__':
    main()