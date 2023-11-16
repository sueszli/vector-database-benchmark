"""A connector for sending API requests to the GCP Recommendations AI
API (https://cloud.google.com/recommendations).
"""
from __future__ import absolute_import
from typing import Sequence
from typing import Tuple
from google.api_core.retry import Retry
from apache_beam import pvalue
from apache_beam.metrics import Metrics
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.transforms import DoFn
from apache_beam.transforms import ParDo
from apache_beam.transforms import PTransform
from apache_beam.transforms.util import GroupIntoBatches
from cachetools.func import ttl_cache
try:
    from google.cloud import recommendationengine
except ImportError:
    raise ImportError('Google Cloud Recommendation AI not supported for this execution environment (could not import google.cloud.recommendationengine).')
__all__ = ['CreateCatalogItem', 'WriteUserEvent', 'ImportCatalogItems', 'ImportUserEvents', 'PredictUserEvent']
FAILED_CATALOG_ITEMS = 'failed_catalog_items'

@ttl_cache(maxsize=128, ttl=3600)
def get_recommendation_prediction_client():
    if False:
        return 10
    'Returns a Recommendation AI - Prediction Service client.'
    _client = recommendationengine.PredictionServiceClient()
    return _client

@ttl_cache(maxsize=128, ttl=3600)
def get_recommendation_catalog_client():
    if False:
        i = 10
        return i + 15
    'Returns a Recommendation AI - Catalog Service client.'
    _client = recommendationengine.CatalogServiceClient()
    return _client

@ttl_cache(maxsize=128, ttl=3600)
def get_recommendation_user_event_client():
    if False:
        while True:
            i = 10
    'Returns a Recommendation AI - UserEvent Service client.'
    _client = recommendationengine.UserEventServiceClient()
    return _client

class CreateCatalogItem(PTransform):
    """Creates catalogitem information.
    The ``PTransform`` returns a PCollectionTuple with a PCollections of
    successfully and failed created CatalogItems.

    Example usage::

      pipeline | CreateCatalogItem(
        project='example-gcp-project',
        catalog_name='my-catalog')
    """

    def __init__(self, project: str=None, retry: Retry=None, timeout: float=120, metadata: Sequence[Tuple[str, str]]=(), catalog_name: str='default_catalog'):
        if False:
            i = 10
            return i + 15
        "Initializes a :class:`CreateCatalogItem` transform.\n\n        Args:\n            project (str): Optional. GCP project name in which the catalog\n              data will be imported.\n            retry: Optional. Designation of what\n              errors, if any, should be retried.\n            timeout (float): Optional. The amount of time, in seconds, to wait\n              for the request to complete.\n            metadata: Optional. Strings which\n              should be sent along with the request as metadata.\n            catalog_name (str): Optional. Name of the catalog.\n              Default: 'default_catalog'\n        "
        self.project = project
        self.retry = retry
        self.timeout = timeout
        self.metadata = metadata
        self.catalog_name = catalog_name

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        if self.project is None:
            self.project = pcoll.pipeline.options.view_as(GoogleCloudOptions).project
        if self.project is None:
            raise ValueError('GCP project name needs to be specified in "project" pipeline\n            option')
        return pcoll | ParDo(_CreateCatalogItemFn(self.project, self.retry, self.timeout, self.metadata, self.catalog_name))

class _CreateCatalogItemFn(DoFn):

    def __init__(self, project: str=None, retry: Retry=None, timeout: float=120, metadata: Sequence[Tuple[str, str]]=(), catalog_name: str=None):
        if False:
            print('Hello World!')
        self._client = None
        self.retry = retry
        self.timeout = timeout
        self.metadata = metadata
        self.parent = f'projects/{project}/locations/global/catalogs/{catalog_name}'
        self.counter = Metrics.counter(self.__class__, 'api_calls')

    def setup(self):
        if False:
            return 10
        if self._client is None:
            self._client = get_recommendation_catalog_client()

    def process(self, element):
        if False:
            for i in range(10):
                print('nop')
        catalog_item = recommendationengine.CatalogItem(element)
        request = recommendationengine.CreateCatalogItemRequest(parent=self.parent, catalog_item=catalog_item)
        try:
            created_catalog_item = self._client.create_catalog_item(request=request, retry=self.retry, timeout=self.timeout, metadata=self.metadata)
            self.counter.inc()
            yield recommendationengine.CatalogItem.to_dict(created_catalog_item)
        except Exception:
            yield pvalue.TaggedOutput(FAILED_CATALOG_ITEMS, recommendationengine.CatalogItem.to_dict(catalog_item))

class ImportCatalogItems(PTransform):
    """Imports catalogitems in bulk.
    The `PTransform` returns a PCollectionTuple with PCollections of
    successfully and failed imported CatalogItems.

    Example usage::

      pipeline
      | ImportCatalogItems(
          project='example-gcp-project',
          catalog_name='my-catalog')
    """

    def __init__(self, max_batch_size: int=5000, project: str=None, retry: Retry=None, timeout: float=120, metadata: Sequence[Tuple[str, str]]=(), catalog_name: str='default_catalog'):
        if False:
            return 10
        "Initializes a :class:`ImportCatalogItems` transform\n\n        Args:\n            batch_size (int): Required. Maximum number of catalogitems per\n              request.\n            project (str): Optional. GCP project name in which the catalog\n              data will be imported.\n            retry: Optional. Designation of what\n              errors, if any, should be retried.\n            timeout (float): Optional. The amount of time, in seconds, to wait\n              for the request to complete.\n            metadata: Optional. Strings which\n              should be sent along with the request as metadata.\n            catalog_name (str): Optional. Name of the catalog.\n              Default: 'default_catalog'\n        "
        self.max_batch_size = max_batch_size
        self.project = project
        self.retry = retry
        self.timeout = timeout
        self.metadata = metadata
        self.catalog_name = catalog_name

    def expand(self, pcoll):
        if False:
            print('Hello World!')
        if self.project is None:
            self.project = pcoll.pipeline.options.view_as(GoogleCloudOptions).project
        if self.project is None:
            raise ValueError('GCP project name needs to be specified in "project" pipeline option')
        return pcoll | GroupIntoBatches.WithShardedKey(self.max_batch_size) | ParDo(_ImportCatalogItemsFn(self.project, self.retry, self.timeout, self.metadata, self.catalog_name))

class _ImportCatalogItemsFn(DoFn):

    def __init__(self, project=None, retry=None, timeout=120, metadata=None, catalog_name=None):
        if False:
            return 10
        self._client = None
        self.retry = retry
        self.timeout = timeout
        self.metadata = metadata
        self.parent = f'projects/{project}/locations/global/catalogs/{catalog_name}'
        self.counter = Metrics.counter(self.__class__, 'api_calls')

    def setup(self):
        if False:
            return 10
        if self._client is None:
            self.client = get_recommendation_catalog_client()

    def process(self, element):
        if False:
            while True:
                i = 10
        catalog_items = [recommendationengine.CatalogItem(e) for e in element[1]]
        catalog_inline_source = recommendationengine.CatalogInlineSource({'catalog_items': catalog_items})
        input_config = recommendationengine.InputConfig(catalog_inline_source=catalog_inline_source)
        request = recommendationengine.ImportCatalogItemsRequest(parent=self.parent, input_config=input_config)
        try:
            operation = self._client.import_catalog_items(request=request, retry=self.retry, timeout=self.timeout, metadata=self.metadata)
            self.counter.inc(len(catalog_items))
            yield operation.result()
        except Exception:
            yield pvalue.TaggedOutput(FAILED_CATALOG_ITEMS, catalog_items)

class WriteUserEvent(PTransform):
    """Write user event information.
    The `PTransform` returns a PCollectionTuple with PCollections of
    successfully and failed written UserEvents.

    Example usage::

      pipeline
      | WriteUserEvent(
          project='example-gcp-project',
          catalog_name='my-catalog',
          event_store='my_event_store')
    """

    def __init__(self, project: str=None, retry: Retry=None, timeout: float=120, metadata: Sequence[Tuple[str, str]]=(), catalog_name: str='default_catalog', event_store: str='default_event_store'):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a :class:`WriteUserEvent` transform.\n\n        Args:\n            project (str): Optional. GCP project name in which the catalog\n              data will be imported.\n            retry: Optional. Designation of what\n              errors, if any, should be retried.\n            timeout (float): Optional. The amount of time, in seconds, to wait\n              for the request to complete.\n            metadata: Optional. Strings which\n              should be sent along with the request as metadata.\n            catalog_name (str): Optional. Name of the catalog.\n              Default: 'default_catalog'\n            event_store (str): Optional. Name of the event store.\n              Default: 'default_event_store'\n        "
        self.project = project
        self.retry = retry
        self.timeout = timeout
        self.metadata = metadata
        self.catalog_name = catalog_name
        self.event_store = event_store

    def expand(self, pcoll):
        if False:
            print('Hello World!')
        if self.project is None:
            self.project = pcoll.pipeline.options.view_as(GoogleCloudOptions).project
        if self.project is None:
            raise ValueError('GCP project name needs to be specified in "project" pipeline option')
        return pcoll | ParDo(_WriteUserEventFn(self.project, self.retry, self.timeout, self.metadata, self.catalog_name, self.event_store))

class _WriteUserEventFn(DoFn):
    FAILED_USER_EVENTS = 'failed_user_events'

    def __init__(self, project=None, retry=None, timeout=120, metadata=None, catalog_name=None, event_store=None):
        if False:
            while True:
                i = 10
        self._client = None
        self.retry = retry
        self.timeout = timeout
        self.metadata = metadata
        self.parent = f'projects/{project}/locations/global/catalogs/{catalog_name}/eventStores/{event_store}'
        self.counter = Metrics.counter(self.__class__, 'api_calls')

    def setup(self):
        if False:
            return 10
        if self._client is None:
            self._client = get_recommendation_user_event_client()

    def process(self, element):
        if False:
            i = 10
            return i + 15
        user_event = recommendationengine.UserEvent(element)
        request = recommendationengine.WriteUserEventRequest(parent=self.parent, user_event=user_event)
        try:
            created_user_event = self._client.write_user_event(request)
            self.counter.inc()
            yield recommendationengine.UserEvent.to_dict(created_user_event)
        except Exception:
            yield pvalue.TaggedOutput(self.FAILED_USER_EVENTS, recommendationengine.UserEvent.to_dict(user_event))

class ImportUserEvents(PTransform):
    """Imports userevents in bulk.
    The `PTransform` returns a PCollectionTuple with PCollections of
    successfully and failed imported UserEvents.

    Example usage::

      pipeline
      | ImportUserEvents(
          project='example-gcp-project',
          catalog_name='my-catalog',
          event_store='my_event_store')
    """

    def __init__(self, max_batch_size: int=5000, project: str=None, retry: Retry=None, timeout: float=120, metadata: Sequence[Tuple[str, str]]=(), catalog_name: str='default_catalog', event_store: str='default_event_store'):
        if False:
            while True:
                i = 10
        "Initializes a :class:`WriteUserEvent` transform.\n\n        Args:\n            batch_size (int): Required. Maximum number of catalogitems\n              per request.\n            project (str): Optional. GCP project name in which the catalog\n              data will be imported.\n            retry: Optional. Designation of what\n              errors, if any, should be retried.\n            timeout (float): Optional. The amount of time, in seconds, to wait\n              for the request to complete.\n            metadata: Optional. Strings which\n              should be sent along with the request as metadata.\n            catalog_name (str): Optional. Name of the catalog.\n              Default: 'default_catalog'\n            event_store (str): Optional. Name of the event store.\n              Default: 'default_event_store'\n        "
        self.max_batch_size = max_batch_size
        self.project = project
        self.retry = retry
        self.timeout = timeout
        self.metadata = metadata
        self.catalog_name = catalog_name
        self.event_store = event_store

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15
        if self.project is None:
            self.project = pcoll.pipeline.options.view_as(GoogleCloudOptions).project
        if self.project is None:
            raise ValueError('GCP project name needs to be specified in "project" pipeline option')
        return pcoll | GroupIntoBatches.WithShardedKey(self.max_batch_size) | ParDo(_ImportUserEventsFn(self.project, self.retry, self.timeout, self.metadata, self.catalog_name, self.event_store))

class _ImportUserEventsFn(DoFn):
    FAILED_USER_EVENTS = 'failed_user_events'

    def __init__(self, project=None, retry=None, timeout=120, metadata=None, catalog_name=None, event_store=None):
        if False:
            print('Hello World!')
        self._client = None
        self.retry = retry
        self.timeout = timeout
        self.metadata = metadata
        self.parent = f'projects/{project}/locations/global/catalogs/{catalog_name}/eventStores/{event_store}'
        self.counter = Metrics.counter(self.__class__, 'api_calls')

    def setup(self):
        if False:
            while True:
                i = 10
        if self._client is None:
            self.client = get_recommendation_user_event_client()

    def process(self, element):
        if False:
            while True:
                i = 10
        user_events = [recommendationengine.UserEvent(e) for e in element[1]]
        user_event_inline_source = recommendationengine.UserEventInlineSource({'user_events': user_events})
        input_config = recommendationengine.InputConfig(user_event_inline_source=user_event_inline_source)
        request = recommendationengine.ImportUserEventsRequest(parent=self.parent, input_config=input_config)
        try:
            operation = self._client.write_user_event(request)
            self.counter.inc(len(user_events))
            yield recommendationengine.PredictResponse.to_dict(operation.result())
        except Exception:
            yield pvalue.TaggedOutput(self.FAILED_USER_EVENTS, user_events)

class PredictUserEvent(PTransform):
    """Make a recommendation prediction.
    The `PTransform` returns a PCollection

    Example usage::

      pipeline
      | PredictUserEvent(
          project='example-gcp-project',
          catalog_name='my-catalog',
          event_store='my_event_store',
          placement_id='recently_viewed_default')
    """

    def __init__(self, project: str=None, retry: Retry=None, timeout: float=120, metadata: Sequence[Tuple[str, str]]=(), catalog_name: str='default_catalog', event_store: str='default_event_store', placement_id: str=None):
        if False:
            while True:
                i = 10
        "Initializes a :class:`PredictUserEvent` transform.\n\n        Args:\n            project (str): Optional. GCP project name in which the catalog\n              data will be imported.\n            retry: Optional. Designation of what\n              errors, if any, should be retried.\n            timeout (float): Optional. The amount of time, in seconds, to wait\n              for the request to complete.\n            metadata: Optional. Strings which\n              should be sent along with the request as metadata.\n            catalog_name (str): Optional. Name of the catalog.\n              Default: 'default_catalog'\n            event_store (str): Optional. Name of the event store.\n              Default: 'default_event_store'\n            placement_id (str): Required. ID of the recommendation engine\n              placement. This id is used to identify the set of models that\n              will be used to make the prediction.\n        "
        self.project = project
        self.retry = retry
        self.timeout = timeout
        self.metadata = metadata
        self.placement_id = placement_id
        self.catalog_name = catalog_name
        self.event_store = event_store
        if placement_id is None:
            raise ValueError('placement_id must be specified')
        else:
            self.placement_id = placement_id

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15
        if self.project is None:
            self.project = pcoll.pipeline.options.view_as(GoogleCloudOptions).project
        if self.project is None:
            raise ValueError('GCP project name needs to be specified in "project" pipeline option')
        return pcoll | ParDo(_PredictUserEventFn(self.project, self.retry, self.timeout, self.metadata, self.catalog_name, self.event_store, self.placement_id))

class _PredictUserEventFn(DoFn):
    FAILED_PREDICTIONS = 'failed_predictions'

    def __init__(self, project=None, retry=None, timeout=120, metadata=None, catalog_name=None, event_store=None, placement_id=None):
        if False:
            while True:
                i = 10
        self._client = None
        self.retry = retry
        self.timeout = timeout
        self.metadata = metadata
        self.name = f'projects/{project}/locations/global/catalogs/{catalog_name}/eventStores/{event_store}/placements/{placement_id}'
        self.counter = Metrics.counter(self.__class__, 'api_calls')

    def setup(self):
        if False:
            return 10
        if self._client is None:
            self._client = get_recommendation_prediction_client()

    def process(self, element):
        if False:
            print('Hello World!')
        user_event = recommendationengine.UserEvent(element)
        request = recommendationengine.PredictRequest(name=self.name, user_event=user_event)
        try:
            prediction = self._client.predict(request)
            self.counter.inc()
            yield [recommendationengine.PredictResponse.to_dict(p) for p in prediction.pages]
        except Exception:
            yield pvalue.TaggedOutput(self.FAILED_PREDICTIONS, user_event)