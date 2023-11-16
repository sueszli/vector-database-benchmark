"""
A connector for sending API requests to the GCP Vision API.
"""
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from apache_beam import typehints
from apache_beam.metrics import Metrics
from apache_beam.transforms import DoFn
from apache_beam.transforms import FlatMap
from apache_beam.transforms import ParDo
from apache_beam.transforms import PTransform
from apache_beam.transforms import util
from cachetools.func import ttl_cache
try:
    from google.cloud import vision
except ImportError:
    raise ImportError('Google Cloud Vision not supported for this execution environment (could not import google.cloud.vision).')
__all__ = ['AnnotateImage', 'AnnotateImageWithContext']

@ttl_cache(maxsize=128, ttl=3600)
def get_vision_client(client_options=None):
    if False:
        while True:
            i = 10
    'Returns a Cloud Vision API client.'
    _client = vision.ImageAnnotatorClient(client_options=client_options)
    return _client

class AnnotateImage(PTransform):
    """A ``PTransform`` for annotating images using the GCP Vision API.
  ref: https://cloud.google.com/vision/docs/

  Batches elements together using ``util.BatchElements`` PTransform and sends
  each batch of elements to the GCP Vision API.
  Element is a Union[str, bytes] of either an URI (e.g. a GCS URI)
  or bytes base64-encoded image data.
  Accepts an `AsDict` side input that maps each image to an image context.
  """
    MAX_BATCH_SIZE = 5
    MIN_BATCH_SIZE = 1

    def __init__(self, features, retry=None, timeout=120, max_batch_size=None, min_batch_size=None, client_options=None, context_side_input=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        '\n    Args:\n      features: (List[``vision.Feature``]) Required.\n        The Vision API features to detect\n      retry: (google.api_core.retry.Retry) Optional.\n        A retry object used to retry requests.\n        If None is specified (default), requests will not be retried.\n      timeout: (float) Optional.\n        The time in seconds to wait for the response from the Vision API.\n        Default is 120.\n      max_batch_size: (int) Optional.\n        Maximum number of images to batch in the same request to the Vision API.\n        Default is 5 (which is also the Vision API max).\n        This parameter is primarily intended for testing.\n      min_batch_size: (int) Optional.\n        Minimum number of images to batch in the same request to the Vision API.\n        Default is None. This parameter is primarily intended for testing.\n      client_options:\n        (Union[dict, google.api_core.client_options.ClientOptions]) Optional.\n        Client options used to set user options on the client.\n        API Endpoint should be set through client_options.\n      context_side_input: (beam.pvalue.AsDict) Optional.\n        An ``AsDict`` of a PCollection to be passed to the\n        _ImageAnnotateFn as the image context mapping containing additional\n        image context and/or feature-specific parameters.\n        Example usage::\n\n          image_contexts =\n            [(\'\'gs://cloud-samples-data/vision/ocr/sign.jpg\'\', Union[dict,\n            ``vision.ImageContext()``]),\n            (\'\'gs://cloud-samples-data/vision/ocr/sign.jpg\'\', Union[dict,\n            ``vision.ImageContext()``]),]\n\n          context_side_input =\n            (\n              p\n              | "Image contexts" >> beam.Create(image_contexts)\n            )\n\n          visionml.AnnotateImage(features,\n            context_side_input=beam.pvalue.AsDict(context_side_input)))\n      metadata: (Optional[Sequence[Tuple[str, str]]]): Optional.\n        Additional metadata that is provided to the method.\n    '
        super().__init__()
        self.features = features
        self.retry = retry
        self.timeout = timeout
        self.max_batch_size = max_batch_size or AnnotateImage.MAX_BATCH_SIZE
        if self.max_batch_size > AnnotateImage.MAX_BATCH_SIZE:
            raise ValueError('Max batch_size exceeded. Batch size needs to be smaller than {}'.format(AnnotateImage.MAX_BATCH_SIZE))
        self.min_batch_size = min_batch_size or AnnotateImage.MIN_BATCH_SIZE
        self.client_options = client_options
        self.context_side_input = context_side_input
        self.metadata = metadata

    def expand(self, pvalue):
        if False:
            i = 10
            return i + 15
        return pvalue | FlatMap(self._create_image_annotation_pairs, self.context_side_input) | util.BatchElements(min_batch_size=self.min_batch_size, max_batch_size=self.max_batch_size) | ParDo(_ImageAnnotateFn(features=self.features, retry=self.retry, timeout=self.timeout, client_options=self.client_options, metadata=self.metadata))

    @typehints.with_input_types(Union[str, bytes], Optional[vision.ImageContext])
    @typehints.with_output_types(List[vision.AnnotateImageRequest])
    def _create_image_annotation_pairs(self, element, context_side_input):
        if False:
            while True:
                i = 10
        if context_side_input:
            image_context = context_side_input.get(element)
        else:
            image_context = None
        if isinstance(element, str):
            image = vision.Image({'source': vision.ImageSource({'image_uri': element})})
        else:
            image = vision.Image(content=element)
        request = vision.AnnotateImageRequest({'image': image, 'features': self.features, 'image_context': image_context})
        yield request

class AnnotateImageWithContext(AnnotateImage):
    """A ``PTransform`` for annotating images using the GCP Vision API.
  ref: https://cloud.google.com/vision/docs/
  Batches elements together using ``util.BatchElements`` PTransform and sends
  each batch of elements to the GCP Vision API.

  Element is a tuple of::

    (Union[str, bytes],
    Optional[``vision.ImageContext``])

  where the former is either an URI (e.g. a GCS URI) or bytes
  base64-encoded image data.
  """

    def __init__(self, features, retry=None, timeout=120, max_batch_size=None, min_batch_size=None, client_options=None, metadata=None):
        if False:
            i = 10
            return i + 15
        '\n    Args:\n      features: (List[``vision.Feature``]) Required.\n        The Vision API features to detect\n      retry: (google.api_core.retry.Retry) Optional.\n        A retry object used to retry requests.\n        If None is specified (default), requests will not be retried.\n      timeout: (float) Optional.\n        The time in seconds to wait for the response from the Vision API.\n        Default is 120.\n      max_batch_size: (int) Optional.\n        Maximum number of images to batch in the same request to the Vision API.\n        Default is 5 (which is also the Vision API max).\n        This parameter is primarily intended for testing.\n      min_batch_size: (int) Optional.\n        Minimum number of images to batch in the same request to the Vision API.\n        Default is None. This parameter is primarily intended for testing.\n      client_options:\n        (Union[dict, google.api_core.client_options.ClientOptions]) Optional.\n        Client options used to set user options on the client.\n        API Endpoint should be set through client_options.\n      metadata: (Optional[Sequence[Tuple[str, str]]]): Optional.\n        Additional metadata that is provided to the method.\n    '
        super().__init__(features=features, retry=retry, timeout=timeout, max_batch_size=max_batch_size, min_batch_size=min_batch_size, client_options=client_options, metadata=metadata)

    def expand(self, pvalue):
        if False:
            i = 10
            return i + 15
        return pvalue | FlatMap(self._create_image_annotation_pairs) | util.BatchElements(min_batch_size=self.min_batch_size, max_batch_size=self.max_batch_size) | ParDo(_ImageAnnotateFn(features=self.features, retry=self.retry, timeout=self.timeout, client_options=self.client_options, metadata=self.metadata))

    @typehints.with_input_types(Tuple[Union[str, bytes], Optional[vision.ImageContext]])
    @typehints.with_output_types(List[vision.AnnotateImageRequest])
    def _create_image_annotation_pairs(self, element, **kwargs):
        if False:
            return 10
        (element, image_context) = element
        if isinstance(element, str):
            image = vision.Image({'source': vision.ImageSource({'image_uri': element})})
        else:
            image = vision.Image({'content': element})
        request = vision.AnnotateImageRequest({'image': image, 'features': self.features, 'image_context': image_context})
        yield request

@typehints.with_input_types(List[vision.AnnotateImageRequest])
class _ImageAnnotateFn(DoFn):
    """A DoFn that sends each input element to the GCP Vision API.
  Returns ``google.cloud.vision.BatchAnnotateImagesResponse``.
  """

    def __init__(self, features, retry, timeout, client_options, metadata):
        if False:
            while True:
                i = 10
        super().__init__()
        self._client = None
        self.features = features
        self.retry = retry
        self.timeout = timeout
        self.client_options = client_options
        self.metadata = metadata
        self.counter = Metrics.counter(self.__class__, 'API Calls')

    def setup(self):
        if False:
            while True:
                i = 10
        self._client = get_vision_client(self.client_options)

    def process(self, element, *args, **kwargs):
        if False:
            while True:
                i = 10
        response = self._client.batch_annotate_images(requests=element, retry=self.retry, timeout=self.timeout, metadata=self.metadata)
        self.counter.inc()
        yield response