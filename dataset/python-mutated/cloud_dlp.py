"""``PTransforms`` that implement Google Cloud Data Loss Prevention
functionality.
"""
import logging
from typing import List
from google.cloud import dlp_v2
from apache_beam import typehints
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.transforms import DoFn
from apache_beam.transforms import ParDo
from apache_beam.transforms import PTransform
__all__ = ['MaskDetectedDetails', 'InspectForDetails']
_LOGGER = logging.getLogger(__name__)

@typehints.with_input_types(str)
@typehints.with_output_types(str)
class MaskDetectedDetails(PTransform):
    """Scrubs sensitive information detected in text.
  The ``PTransform`` returns a ``PCollection`` of ``str``
  Example usage::

    pipeline | MaskDetectedDetails(project='example-gcp-project',
      deidentification_config={
          'info_type_transformations: {
              'transformations': [{
                  'primitive_transformation': {
                      'character_mask_config': {
                          'masking_character': '#'
                      }
                  }
              }]
          }
      }, inspection_config={'info_types': [{'name': 'EMAIL_ADDRESS'}]})

  """

    def __init__(self, project=None, deidentification_template_name=None, deidentification_config=None, inspection_template_name=None, inspection_config=None, timeout=None):
        if False:
            i = 10
            return i + 15
        'Initializes a :class:`MaskDetectedDetails` transform.\n\n    Args:\n      project: Optional. GCP project name in which inspection will be performed\n      deidentification_template_name (str): Either this or\n        `deidentification_config` required. Name of\n        deidentification template to be used on detected sensitive information\n        instances in text.\n      deidentification_config\n        (``Union[dict, google.cloud.dlp_v2.types.DeidentifyConfig]``):\n        Configuration for the de-identification of the content item.\n        If both template name and config are supplied,\n        config is more important.\n      inspection_template_name (str): This or `inspection_config` required.\n        Name of inspection template to be used\n        to detect sensitive data in text.\n      inspection_config\n        (``Union[dict, google.cloud.dlp_v2.types.InspectConfig]``):\n        Configuration for the inspector used to detect sensitive data in text.\n        If both template name and config are supplied,\n        config takes precedence.\n      timeout (float): Optional. The amount of time, in seconds, to wait for\n        the request to complete.\n\n    '
        self.config = {}
        self.project = project
        self.timeout = timeout
        if deidentification_template_name is not None and deidentification_config is not None:
            raise ValueError('Both deidentification_template_name and deidentification_config were specified. Please specify only one of these.')
        elif deidentification_template_name is None and deidentification_config is None:
            raise ValueError('deidentification_template_name or deidentification_config must be specified.')
        elif deidentification_template_name is not None:
            self.config['deidentify_template_name'] = deidentification_template_name
        else:
            self.config['deidentify_config'] = deidentification_config
        if inspection_config is None and inspection_template_name is None:
            raise ValueError('inspection_template_name or inspection_config must be specified')
        if inspection_template_name is not None:
            self.config['inspect_template_name'] = inspection_template_name
        if inspection_config is not None:
            self.config['inspect_config'] = inspection_config

    def expand(self, pcoll):
        if False:
            print('Hello World!')
        if self.project is None:
            self.project = pcoll.pipeline.options.view_as(GoogleCloudOptions).project
        if self.project is None:
            raise ValueError('GCP project name needs to be specified in "project" pipeline option')
        return pcoll | ParDo(_DeidentifyFn(self.config, self.timeout, self.project))

@typehints.with_input_types(str)
@typehints.with_output_types(List[dlp_v2.types.dlp.Finding])
class InspectForDetails(PTransform):
    """Inspects input text for sensitive information.
  the ``PTransform`` returns a ``PCollection`` of
  ``List[google.cloud.dlp_v2.proto.dlp_pb2.Finding]``
  Example usage::

      pipeline | InspectForDetails(project='example-gcp-project',
                inspection_config={'info_types': [{'name': 'EMAIL_ADDRESS'}]})
  """

    def __init__(self, project=None, inspection_template_name=None, inspection_config=None, timeout=None):
        if False:
            i = 10
            return i + 15
        'Initializes a :class:`InspectForDetails` transform.\n\n    Args:\n      project: Optional. GCP project name in which inspection will be performed\n      inspection_template_name (str): This or `inspection_config` required.\n        Name of inspection template to be used\n        to detect sensitive data in text.\n      inspection_config\n        (``Union[dict, google.cloud.dlp_v2.types.InspectConfig]``):\n        Configuration for the inspector used to detect sensitive data in text.\n        If both template name and config are supplied,\n        config takes precedence.\n      timeout (float): Optional. The amount of time, in seconds, to wait for\n        the request to complete.\n\n    '
        self.timeout = timeout
        self.config = {}
        self.project = project
        if inspection_config is None and inspection_template_name is None:
            raise ValueError('inspection_template_name or inspection_config must be specified')
        if inspection_template_name is not None:
            self.config['inspect_template_name'] = inspection_template_name
        if inspection_config is not None:
            self.config['inspect_config'] = inspection_config

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        if self.project is None:
            self.project = pcoll.pipeline.options.view_as(GoogleCloudOptions).project
        if self.project is None:
            raise ValueError('GCP project name needs to be specified in "project" pipeline option')
        return pcoll | ParDo(_InspectFn(self.config, self.timeout, self.project))

class _DeidentifyFn(DoFn):

    def __init__(self, config=None, timeout=None, project=None, client=None):
        if False:
            return 10
        self.config = config
        self.timeout = timeout
        self.client = client
        self.project = project
        self.params = {}

    def setup(self):
        if False:
            return 10
        if self.client is None:
            self.client = dlp_v2.DlpServiceClient()
        self.params = {'timeout': self.timeout}
        self.parent = self.client.common_project_path(self.project)

    def process(self, element, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        request = {'item': {'value': element}, 'parent': self.parent}
        request.update(self.config)
        operation = self.client.deidentify_content(request=request, **self.params)
        yield operation.item.value

class _InspectFn(DoFn):

    def __init__(self, config=None, timeout=None, project=None):
        if False:
            while True:
                i = 10
        self.config = config
        self.timeout = timeout
        self.client = None
        self.project = project
        self.params = {}

    def setup(self):
        if False:
            while True:
                i = 10
        if self.client is None:
            self.client = dlp_v2.DlpServiceClient()
        self.params = {'timeout': self.timeout}
        self.parent = self.client.common_project_path(self.project)

    def process(self, element, **kwargs):
        if False:
            i = 10
            return i + 15
        request = {'item': {'value': element}, 'parent': self.parent}
        request.update(self.config)
        operation = self.client.inspect_content(request=request, **self.params)
        hits = [x for x in operation.result.findings]
        yield hits