from __future__ import annotations
import pytest
from airflow.api_connexion.endpoints.update_mask import extract_update_mask_data
from airflow.api_connexion.exceptions import BadRequest

class TestUpdateMask:

    def test_should_extract_data(self):
        if False:
            print('Hello World!')
        non_update_fields = ['field_1']
        update_mask = ['field_2']
        data = {'field_1': 'value_1', 'field_2': 'value_2', 'field_3': 'value_3'}
        data = extract_update_mask_data(update_mask, non_update_fields, data)
        assert data == {'field_2': 'value_2'}

    def test_update_forbid_field_should_raise_exception(self):
        if False:
            return 10
        non_update_fields = ['field_1']
        update_mask = ['field_1', 'field_2']
        data = {'field_1': 'value_1', 'field_2': 'value_2', 'field_3': 'value_3'}
        with pytest.raises(BadRequest):
            extract_update_mask_data(update_mask, non_update_fields, data)

    def test_update_unknown_field_should_raise_exception(self):
        if False:
            print('Hello World!')
        non_update_fields = ['field_1']
        update_mask = ['field_2', 'field_3']
        data = {'field_1': 'value_1', 'field_2': 'value_2'}
        with pytest.raises(BadRequest):
            extract_update_mask_data(update_mask, non_update_fields, data)