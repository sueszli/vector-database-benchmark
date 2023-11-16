from __future__ import annotations
from airflow.api_connexion.schemas.config_schema import Config, ConfigOption, ConfigSection, config_schema

class TestConfigSchema:

    def test_serialize(self):
        if False:
            for i in range(10):
                print('nop')
        config = Config(sections=[ConfigSection(name='sec1', options=[ConfigOption(key='apache', value='airflow'), ConfigOption(key='hello', value='world')]), ConfigSection(name='sec2', options=[ConfigOption(key='foo', value='bar')])])
        result = config_schema.dump(config)
        expected = {'sections': [{'name': 'sec1', 'options': [{'key': 'apache', 'value': 'airflow'}, {'key': 'hello', 'value': 'world'}]}, {'name': 'sec2', 'options': [{'key': 'foo', 'value': 'bar'}]}]}
        assert result == expected