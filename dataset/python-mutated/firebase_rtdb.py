import json
import firebase_admin
from firebase_admin import credentials, db

class Client:

    def __init__(self, path='', buffer_size=10000):
        if False:
            for i in range(10):
                print('nop')
        self._path = path
        self._buffer_size = buffer_size

    def initialize(self, database_name, google_application_credentials):
        if False:
            i = 10
            return i + 15
        database_url = f'https://{database_name}.firebaseio.com'
        sa_key = json.loads(google_application_credentials)
        cred = credentials.Certificate(sa_key)
        firebase_admin.initialize_app(cred, {'databaseURL': database_url})
        self._ref = db.reference(self._path)

    def check_connection(self):
        if False:
            while True:
                i = 10
        self._ref.get(shallow=True)

    def fetch_records(self, start_key=None):
        if False:
            for i in range(10):
                print('nop')
        if start_key:
            return self._ref.order_by_key().start_at(start_key).limit_to_first(self._buffer_size).get()
        else:
            return self._ref.order_by_key().limit_to_first(self._buffer_size).get()

    def extract(self):
        if False:
            i = 10
            return i + 15
        return Records(self)

    def set_records(self, records):
        if False:
            return 10
        self._ref.set(records)

    def delete_records(self):
        if False:
            for i in range(10):
                print('nop')
        self._ref.delete()

class Records:

    def __init__(self, client):
        if False:
            print('Hello World!')
        self._client = client

    def __iter__(self):
        if False:
            print('Hello World!')

        def _gen():
            if False:
                i = 10
                return i + 15
            records = self._client.fetch_records()
            if records is None or len(records) == 0:
                return
            for (k, v) in records.items():
                last_key = k
                data = {'key': k, 'value': json.dumps(v)}
                yield data
            while (records := self._client.fetch_records(last_key)):
                num_records = len(records)
                records_iter = iter(records.items())
                (first_key, first_value) = next(records_iter)
                if first_key == last_key:
                    if num_records == 1:
                        return
                else:
                    last_key = first_key
                    data = {'key': first_key, 'value': json.dumps(first_value)}
                    yield data
                for (k, v) in records_iter:
                    last_key = k
                    data = {'key': k, 'value': json.dumps(v)}
                    yield data
        return _gen()