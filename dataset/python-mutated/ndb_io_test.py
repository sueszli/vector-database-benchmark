"""Unit tests for jobs.io.ndb_io."""
from __future__ import annotations
from core.jobs import job_test_utils
from core.jobs.io import ndb_io
from core.platform import models
import apache_beam as beam
from typing import List
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import datastore_services
(base_models,) = models.Registry.import_models([models.Names.BASE_MODEL])
datastore_services = models.Registry.import_datastore_services()

class NdbIoTests(job_test_utils.PipelinedTestBase):

    def get_base_models(self) -> List[base_models.BaseModel]:
        if False:
            print('Hello World!')
        'Returns all models in the datastore.\n\n        Returns:\n            list(Model). All of the models in the datastore.\n        '
        return list(base_models.BaseModel.get_all())

    def put_multi(self, model_list: List[base_models.BaseModel]) -> None:
        if False:
            i = 10
            return i + 15
        'Puts the given models into the datastore.\n\n        Args:\n            model_list: list(Model). The models to put into the datastore.\n        '
        datastore_services.update_timestamps_multi(model_list, update_last_updated_time=False)
        datastore_services.put_multi(model_list)

    def test_read_from_datastore(self) -> None:
        if False:
            while True:
                i = 10
        model_list = [self.create_model(base_models.BaseModel, id='a'), self.create_model(base_models.BaseModel, id='b'), self.create_model(base_models.BaseModel, id='c')]
        self.put_multi(model_list)
        self.assertItemsEqual(self.get_base_models(), model_list)
        model_pcoll = self.pipeline | ndb_io.GetModels(base_models.BaseModel.get_all())
        self.assert_pcoll_equal(model_pcoll, model_list)

    def test_write_to_datastore(self) -> None:
        if False:
            print('Hello World!')
        model_list = [self.create_model(base_models.BaseModel, id='a'), self.create_model(base_models.BaseModel, id='b'), self.create_model(base_models.BaseModel, id='c')]
        self.assertItemsEqual(self.get_base_models(), [])
        self.assert_pcoll_empty(self.pipeline | beam.Create(model_list) | ndb_io.PutModels())
        self.assertItemsEqual(self.get_base_models(), model_list)

    def test_delete_from_datastore(self) -> None:
        if False:
            print('Hello World!')
        model_list = [self.create_model(base_models.BaseModel, id='a'), self.create_model(base_models.BaseModel, id='b'), self.create_model(base_models.BaseModel, id='c')]
        self.put_multi(model_list)
        self.assertItemsEqual(self.get_base_models(), model_list)
        self.assert_pcoll_empty(self.pipeline | beam.Create([model.key for model in model_list]) | ndb_io.DeleteModels())
        self.assertItemsEqual(self.get_base_models(), [])