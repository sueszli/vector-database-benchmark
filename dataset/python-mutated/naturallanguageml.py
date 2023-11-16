from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import apache_beam as beam
from apache_beam.metrics import Metrics
try:
    from google.cloud import language
    from google.cloud import language_v1
except ImportError:
    raise ImportError('Google Cloud Natural Language API not supported for this execution environment (could not import Natural Language API client).')
__all__ = ['Document', 'AnnotateText']

class Document(object):
    """Represents the input to :class:`AnnotateText` transform.

  Args:
    content (str): The content of the input or the Google Cloud Storage URI
      where the file is stored.
    type (`Union[str, google.cloud.language_v1.Document.Type]`): Text type.
      Possible values are `HTML`, `PLAIN_TEXT`. The default value is
      `PLAIN_TEXT`.
    language_hint (`Optional[str]`): The language of the text. If not specified,
      language will be automatically detected. Values should conform to
      ISO-639-1 standard.
    encoding (`Optional[str]`): Text encoding. Possible values are: `NONE`,
     `UTF8`, `UTF16`, `UTF32`. The default value is `UTF8`.
    from_gcs (bool): Whether the content should be interpret as a Google Cloud
      Storage URI. The default value is :data:`False`.
  """

    def __init__(self, content, type='PLAIN_TEXT', language_hint=None, encoding='UTF8', from_gcs=False):
        if False:
            for i in range(10):
                print('nop')
        self.content = content
        self.type = type
        self.encoding = encoding
        self.language_hint = language_hint
        self.from_gcs = from_gcs

    @staticmethod
    def to_dict(document):
        if False:
            return 10
        if document.from_gcs:
            dict_repr = {'gcs_content_uri': document.content}
        else:
            dict_repr = {'content': document.content}
        dict_repr.update({'type': document.type, 'language': document.language_hint})
        return dict_repr

@beam.ptransform_fn
def AnnotateText(pcoll, features, timeout=None, metadata=None):
    if False:
        while True:
            i = 10
    "A :class:`~apache_beam.transforms.ptransform.PTransform`\n  for annotating text using the Google Cloud Natural Language API:\n  https://cloud.google.com/natural-language/docs.\n\n  Args:\n    pcoll (:class:`~apache_beam.pvalue.PCollection`): An input PCollection of\n      :class:`Document` objects.\n    features (`Union[Mapping[str, bool], types.AnnotateTextRequest.Features]`):\n      A dictionary of natural language operations to be performed on given\n      text in the following format::\n      {'extact_syntax'=True, 'extract_entities'=True}\n\n    timeout (`Optional[float]`): The amount of time, in seconds, to wait\n      for the request to complete. The timeout applies to each individual\n      retry attempt.\n    metadata (`Optional[Sequence[Tuple[str, str]]]`): Additional metadata\n      that is provided to the method.\n  "
    return pcoll | beam.ParDo(_AnnotateTextFn(features, timeout, metadata))

@beam.typehints.with_input_types(Document)
@beam.typehints.with_output_types(language_v1.AnnotateTextResponse)
class _AnnotateTextFn(beam.DoFn):

    def __init__(self, features, timeout, metadata=None):
        if False:
            print('Hello World!')
        self.features = features
        self.timeout = timeout
        self.metadata = metadata
        self.api_calls = Metrics.counter(self.__class__.__name__, 'api_calls')
        self.client = None

    def setup(self):
        if False:
            while True:
                i = 10
        self.client = self._get_api_client()

    @staticmethod
    def _get_api_client():
        if False:
            for i in range(10):
                print('nop')
        return language.LanguageServiceClient()

    def process(self, element):
        if False:
            i = 10
            return i + 15
        response = self.client.annotate_text(document=Document.to_dict(element), features=self.features, encoding_type=element.encoding, timeout=self.timeout, metadata=self.metadata)
        self.api_calls.inc()
        yield response