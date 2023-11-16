"""Metadata for use in BigQueryIO, i.e. a job_id to use in BQ job labels."""
import re
from apache_beam.io.gcp import gce_metadata_util
_VALID_CLOUD_LABEL_PATTERN = re.compile('^[a-z0-9\\_\\-]{1,63}$')

def _sanitize_value(value):
    if False:
        return 10
    'Sanitizes a value into a valid BigQuery label value.'
    return re.sub('[^\\w-]+', '', value.lower().replace('/', '-'))[0:63]

def _is_valid_cloud_label_value(label_value):
    if False:
        while True:
            i = 10
    'Returns true if label_value is a valid cloud label string.\n\n    This function can return false in cases where the label value is valid.\n    However, it will not return true in a case where the lavel value is invalid.\n    This is because a stricter set of allowed characters is used in this\n    validator, because foreign language characters are not accepted.\n    Thus, this should not be used as a generic validator for all cloud labels.\n\n    See Also:\n      https://cloud.google.com/compute/docs/labeling-resources\n\n    Args:\n      label_value: The label value to validate.\n\n    Returns:\n      True if the label value is a valid\n  '
    return _VALID_CLOUD_LABEL_PATTERN.match(label_value)

def create_bigquery_io_metadata(step_name=None):
    if False:
        while True:
            i = 10
    'Creates a BigQueryIOMetadata.\n\n  This will request metadata properly based on which runner is being used.\n  '
    dataflow_job_id = gce_metadata_util.fetch_dataflow_job_id()
    is_dataflow_runner = bool(dataflow_job_id)
    kwargs = {}
    if is_dataflow_runner:
        if _is_valid_cloud_label_value(dataflow_job_id):
            kwargs['beam_job_id'] = dataflow_job_id
    if step_name:
        step_name = _sanitize_value(step_name)
        if _is_valid_cloud_label_value(step_name):
            kwargs['step_name'] = step_name
    return BigQueryIOMetadata(**kwargs)

class BigQueryIOMetadata(object):
    """Metadata class for BigQueryIO. i.e. to use as BQ job labels.

  Do not construct directly, use the create_bigquery_io_metadata factory.
  Which will request metadata properly based on which runner is being used.
  """

    def __init__(self, beam_job_id=None, step_name=None):
        if False:
            while True:
                i = 10
        self.beam_job_id = beam_job_id
        self.step_name = step_name

    def add_additional_bq_job_labels(self, job_labels=None):
        if False:
            print('Hello World!')
        job_labels = job_labels or {}
        if self.beam_job_id and 'beam_job_id' not in job_labels:
            job_labels['beam_job_id'] = self.beam_job_id
        if self.step_name and 'step_name' not in job_labels:
            job_labels['step_name'] = self.step_name
        return job_labels