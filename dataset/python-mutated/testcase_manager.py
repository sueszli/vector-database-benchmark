from seleniumbase.core.mysql import DatabaseManager

class TestcaseManager:

    def __init__(self, database_env):
        if False:
            print('Hello World!')
        self.database_env = database_env

    def insert_execution_data(self, execution_query_payload):
        if False:
            for i in range(10):
                print('nop')
        'Inserts a test execution row into the database.\n        Returns the execution guid.\n        "execution_start_time" is defined by milliseconds since the Epoch.\n        (See https://currentmillis.com to convert that to a real date.)'
        query = 'INSERT INTO test_execution\n                   (guid, execution_start, total_execution_time, username)\n                   VALUES (%(guid)s,%(execution_start_time)s,\n                           %(total_execution_time)s,%(username)s)'
        DatabaseManager(self.database_env).execute_query(query, execution_query_payload.get_params())
        return execution_query_payload.guid

    def update_execution_data(self, execution_guid, execution_time):
        if False:
            return 10
        'Updates an existing test execution row in the database.'
        query = 'UPDATE test_execution\n                   SET total_execution_time=%(execution_time)s\n                   WHERE guid=%(execution_guid)s '
        DatabaseManager(self.database_env).execute_query(query, {'execution_guid': execution_guid, 'execution_time': execution_time})

    def insert_testcase_data(self, testcase_run_payload):
        if False:
            print('Hello World!')
        'Inserts all data for the test in the DB. Returns new row guid.'
        query = 'INSERT INTO test_run_data(\n                   guid, browser, state, execution_guid, env, start_time,\n                   test_address, runtime, retry_count, message, stack_trace)\n                          VALUES (\n                              %(guid)s,\n                              %(browser)s,\n                              %(state)s,\n                              %(execution_guid)s,\n                              %(env)s,\n                              %(start_time)s,\n                              %(test_address)s,\n                              %(runtime)s,\n                              %(retry_count)s,\n                              %(message)s,\n                              %(stack_trace)s) '
        DatabaseManager(self.database_env).execute_query(query, testcase_run_payload.get_params())

    def update_testcase_data(self, testcase_payload):
        if False:
            while True:
                i = 10
        'Updates an existing test run in the database.'
        query = 'UPDATE test_run_data SET\n                            runtime=%(runtime)s,\n                            state=%(state)s,\n                            retry_count=%(retry_count)s,\n                            stack_trace=%(stack_trace)s,\n                            message=%(message)s\n                            WHERE guid=%(guid)s '
        DatabaseManager(self.database_env).execute_query(query, testcase_payload.get_params())

    def update_testcase_log_url(self, testcase_payload):
        if False:
            return 10
        query = 'UPDATE test_run_data\n                   SET log_url=%(log_url)s\n                   WHERE guid=%(guid)s '
        DatabaseManager(self.database_env).execute_query(query, testcase_payload.get_params())

class ExecutionQueryPayload:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.execution_start_time = None
        self.total_execution_time = -1
        self.username = 'Default'
        self.guid = None

    def get_params(self):
        if False:
            return 10
        return {'execution_start_time': self.execution_start_time, 'total_execution_time': self.total_execution_time, 'username': self.username, 'guid': self.guid}

class TestcaseDataPayload:

    def __init__(self):
        if False:
            print('Hello World!')
        self.guid = None
        self.test_address = None
        self.browser = None
        self.state = None
        self.execution_guid = None
        self.env = None
        self.start_time = None
        self.runtime = None
        self.retry_count = 0
        self.stack_trace = None
        self.message = None
        self.log_url = None

    def get_params(self):
        if False:
            print('Hello World!')
        return {'guid': self.guid, 'test_address': self.test_address, 'browser': self.browser, 'state': self.state, 'execution_guid': self.execution_guid, 'env': self.env, 'start_time': self.start_time, 'runtime': self.runtime, 'retry_count': self.retry_count, 'stack_trace': self.stack_trace, 'message': self.message, 'log_url': self.log_url}