from __future__ import annotations
from airflow.api_connexion.schemas.health_schema import health_schema

class TestHealthSchema:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.default_datetime = '2020-06-10T12:02:44+00:00'

    def test_serialize(self):
        if False:
            while True:
                i = 10
        payload = {'metadatabase': {'status': 'healthy'}, 'scheduler': {'status': 'healthy', 'latest_scheduler_heartbeat': self.default_datetime}}
        serialized_data = health_schema.dump(payload)
        assert serialized_data == payload