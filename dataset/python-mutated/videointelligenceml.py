"""A connector for sending API requests to the GCP Video Intelligence API."""
from typing import Optional
from typing import Tuple
from typing import Union
from apache_beam import typehints
from apache_beam.metrics import Metrics
from apache_beam.transforms import DoFn
from apache_beam.transforms import ParDo
from apache_beam.transforms import PTransform
from cachetools.func import ttl_cache
try:
    from google.cloud import videointelligence
except ImportError:
    raise ImportError('Google Cloud Video Intelligence not supported for this execution environment (could not import google.cloud.videointelligence).')
__all__ = ['AnnotateVideo', 'AnnotateVideoWithContext']

@ttl_cache(maxsize=128, ttl=3600)
def get_videointelligence_client():
    if False:
        i = 10
        return i + 15
    'Returns a Cloud Video Intelligence client.'
    _client = videointelligence.VideoIntelligenceServiceClient()
    return _client

class AnnotateVideo(PTransform):
    """A ``PTransform`` for annotating video using the GCP Video Intelligence API
  ref: https://cloud.google.com/video-intelligence/docs

  Sends each element to the GCP Video Intelligence API. Element is a
  Union[str, bytes] of either an URI (e.g. a GCS URI) or
  bytes base64-encoded video data.
  Accepts an `AsDict` side input that maps each video to a video context.
  """

    def __init__(self, features, location_id=None, metadata=None, timeout=120, context_side_input=None):
        if False:
            while True:
                i = 10
        '\n    Args:\n      features: (List[``videointelligence_v1.Feature``]) Required.\n        The Video Intelligence API features to detect\n      location_id: (str) Optional.\n        Cloud region where annotation should take place.\n        If no region is specified, a region will be determined\n        based on video file location.\n      metadata: (Sequence[Tuple[str, str]]) Optional.\n        Additional metadata that is provided to the method.\n      timeout: (int) Optional.\n        The time in seconds to wait for the response from the\n        Video Intelligence API\n      context_side_input: (beam.pvalue.AsDict) Optional.\n        An ``AsDict`` of a PCollection to be passed to the\n        _VideoAnnotateFn as the video context mapping containing additional\n        video context and/or feature-specific parameters.\n        Example usage::\n\n          video_contexts =\n            [(\'gs://cloud-samples-data/video/cat.mp4\', Union[dict,\n            ``videointelligence_v1.VideoContext``]),\n            (\'gs://some-other-video/sample.mp4\', Union[dict,\n            ``videointelligence_v1.VideoContext``]),]\n\n          context_side_input =\n            (\n              p\n              | "Video contexts" >> beam.Create(video_contexts)\n            )\n\n          videointelligenceml.AnnotateVideo(features,\n            context_side_input=beam.pvalue.AsDict(context_side_input)))\n    '
        super().__init__()
        self.features = features
        self.location_id = location_id
        self.metadata = metadata
        self.timeout = timeout
        self.context_side_input = context_side_input

    def expand(self, pvalue):
        if False:
            print('Hello World!')
        return pvalue | ParDo(_VideoAnnotateFn(features=self.features, location_id=self.location_id, metadata=self.metadata, timeout=self.timeout), context_side_input=self.context_side_input)

@typehints.with_input_types(Union[str, bytes], Optional[videointelligence.VideoContext])
class _VideoAnnotateFn(DoFn):
    """A DoFn that sends each input element to the GCP Video Intelligence API
  service and outputs an element with the return result of the API
  (``google.cloud.videointelligence_v1.AnnotateVideoResponse``).
  """

    def __init__(self, features, location_id, metadata, timeout):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._client = None
        self.features = features
        self.location_id = location_id
        self.metadata = metadata
        self.timeout = timeout
        self.counter = Metrics.counter(self.__class__, 'API Calls')

    def start_bundle(self):
        if False:
            return 10
        self._client = get_videointelligence_client()

    def _annotate_video(self, element, video_context):
        if False:
            while True:
                i = 10
        if isinstance(element, str):
            response = self._client.annotate_video(input_uri=element, features=self.features, video_context=video_context, location_id=self.location_id, metadata=self.metadata)
        else:
            response = self._client.annotate_video(input_content=element, features=self.features, video_context=video_context, location_id=self.location_id, metadata=self.metadata)
        return response

    def process(self, element, context_side_input=None, *args, **kwargs):
        if False:
            print('Hello World!')
        if context_side_input:
            video_context = context_side_input.get(element)
        else:
            video_context = None
        response = self._annotate_video(element, video_context)
        self.counter.inc()
        yield response.result(timeout=self.timeout)

class AnnotateVideoWithContext(AnnotateVideo):
    """A ``PTransform`` for annotating video using the GCP Video Intelligence API
  ref: https://cloud.google.com/video-intelligence/docs

  Sends each element to the GCP Video Intelligence API.
  Element is a tuple of

    (Union[str, bytes],
    Optional[videointelligence.VideoContext])

  where the former is either an URI (e.g. a GCS URI) or
  bytes base64-encoded video data
  """

    def __init__(self, features, location_id=None, metadata=None, timeout=120):
        if False:
            print('Hello World!')
        '\n      Args:\n        features: (List[``videointelligence_v1.Feature``]) Required.\n          the Video Intelligence API features to detect\n        location_id: (str) Optional.\n          Cloud region where annotation should take place.\n          If no region is specified, a region will be determined\n          based on video file location.\n        metadata: (Sequence[Tuple[str, str]]) Optional.\n          Additional metadata that is provided to the method.\n        timeout: (int) Optional.\n          The time in seconds to wait for the response from the\n          Video Intelligence API\n    '
        super().__init__(features=features, location_id=location_id, metadata=metadata, timeout=timeout)

    def expand(self, pvalue):
        if False:
            i = 10
            return i + 15
        return pvalue | ParDo(_VideoAnnotateFnWithContext(features=self.features, location_id=self.location_id, metadata=self.metadata, timeout=self.timeout))

@typehints.with_input_types(Tuple[Union[str, bytes], Optional[videointelligence.VideoContext]])
class _VideoAnnotateFnWithContext(_VideoAnnotateFn):
    """A DoFn that unpacks each input tuple to element, video_context variables
  and sends these to the GCP Video Intelligence API service and outputs
  an element with the return result of the API
  (``google.cloud.videointelligence_v1.AnnotateVideoResponse``).
  """

    def __init__(self, features, location_id, metadata, timeout):
        if False:
            i = 10
            return i + 15
        super().__init__(features=features, location_id=location_id, metadata=metadata, timeout=timeout)

    def process(self, element, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        (element, video_context) = element
        response = self._annotate_video(element, video_context)
        self.counter.inc()
        yield response.result(timeout=self.timeout)