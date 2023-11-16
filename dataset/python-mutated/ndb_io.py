"""Provides an Apache Beam API for operating on NDB models."""
from __future__ import annotations
from core import feconf
from core.jobs import job_utils
from core.platform import models
import apache_beam as beam
from apache_beam import pvalue
from apache_beam.io.gcp.datastore.v1new import datastoreio
from typing import Optional
MYPY = False
if MYPY:
    from mypy_imports import datastore_services
datastore_services = models.Registry.import_datastore_services()

class GetModels(beam.PTransform):
    """Reads NDB models from the datastore using a query."""

    def __init__(self, query: datastore_services.Query, label: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Initializes the GetModels PTransform.\n\n        Args:\n            query: datastore_services.Query. The query used to fetch models.\n            label: str|None. The label of the PTransform.\n        '
        super().__init__(label=label)
        self.query = query

    def expand(self, pbegin: pvalue.PBegin) -> beam.PCollection[datastore_services.Model]:
        if False:
            return 10
        'Returns a PCollection with models matching the corresponding query.\n\n        This overrides the expand() method from the parent class.\n\n        Args:\n            pbegin: PValue. The initial pipeline. This pipeline\n                is used to anchor the models to itself.\n\n        Returns:\n            PCollection. The PCollection of models.\n        '
        query = job_utils.get_beam_query_from_ndb_query(self.query, namespace=pbegin.pipeline.options.namespace)
        return pbegin.pipeline | 'Reading %r from the datastore' % self.query >> datastoreio.ReadFromDatastore(query) | 'Transforming %r into NDB models' % self.query >> beam.Map(job_utils.get_ndb_model_from_beam_entity)

class PutModels(beam.PTransform):
    """Writes NDB models to the datastore."""

    def expand(self, entities: beam.PCollection[datastore_services.Model]) -> pvalue.PDone:
        if False:
            while True:
                i = 10
        'Writes the given models to the datastore.\n\n        This overrides the expand() method from the parent class.\n\n        Args:\n            entities: PCollection. A PCollection of NDB models to write\n                to the datastore. Can also contain just one model.\n\n        Returns:\n            PCollection. An empty PCollection. This is needed because all\n            expand() methods need to return some PCollection.\n        '
        return entities | 'Transforming the NDB models into Apache Beam entities' >> beam.Map(job_utils.get_beam_entity_from_ndb_model) | 'Writing the NDB models to the datastore' >> datastoreio.WriteToDatastore(feconf.OPPIA_PROJECT_ID)

class DeleteModels(beam.PTransform):
    """Deletes NDB models from the datastore."""

    def expand(self, entities: beam.PCollection[datastore_services.Key]) -> pvalue.PDone:
        if False:
            while True:
                i = 10
        'Deletes the given models from the datastore.\n\n        This overrides the expand() method from the parent class.\n\n        Args:\n            entities: PCollection. The PCollection of NDB keys to delete\n                from the datastore. Can also contain just one model.\n\n        Returns:\n            PCollection. An empty PCollection. This is needed because all\n            expand() methods need to return some PCollection.\n        '
        return entities | 'Transforming the NDB keys into Apache Beam keys' >> beam.Map(job_utils.get_beam_key_from_ndb_key) | 'Deleting the NDB keys from the datastore' >> datastoreio.DeleteFromDatastore(feconf.OPPIA_PROJECT_ID)