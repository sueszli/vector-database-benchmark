"""Wraps the Google Cloud Storage client library for use in tables helper."""
import logging
import time
from google.api_core import exceptions
try:
    import pandas
except ImportError:
    pandas = None
try:
    from google.cloud import storage
except ImportError:
    storage = None
_LOGGER = logging.getLogger(__name__)
_PANDAS_REQUIRED = 'pandas is required to verify type DataFrame.'
_STORAGE_REQUIRED = 'google-cloud-storage is required to create a Google Cloud Storage client.'

class GcsClient(object):
    """Uploads Pandas DataFrame to a bucket in Google Cloud Storage."""

    def __init__(self, bucket_name=None, client=None, credentials=None, project=None):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n        Args:\n            bucket_name (Optional[str]): The name of Google Cloud Storage\n                bucket for this client to send requests to.\n            client (Optional[storage.Client]): A Google Cloud Storage Client\n                instance.\n            credentials (Optional[google.auth.credentials.Credentials]): The\n                authorization credentials to attach to requests. These\n                credentials identify this application to the service. If none\n                are specified, the client will attempt to ascertain the\n                credentials from the environment.\n            project (Optional[str]): The project ID of the GCP project to\n                attach to the underlying storage client. If none is specified,\n                the client will attempt to ascertain the credentials from the\n                environment.\n        '
        if storage is None:
            raise ImportError(_STORAGE_REQUIRED)
        if client is not None:
            self.client = client
        elif credentials is not None:
            self.client = storage.Client(credentials=credentials, project=project)
        else:
            self.client = storage.Client()
        self.bucket_name = bucket_name

    def ensure_bucket_exists(self, project, region):
        if False:
            for i in range(10):
                print('nop')
        "Checks if a bucket named '{project}-automl-tables-staging' exists.\n\n        If this bucket doesn't exist, creates one.\n        If this bucket already exists in `project`, do nothing.\n        If this bucket exists in a different project that we don't have\n        access to, creates a bucket named\n        '{project}-automl-tables-staging-{create_timestamp}' because bucket's\n        name must be globally unique.\n        Save the created bucket's name and reuse this for future requests.\n\n        Args:\n            project (str): The ID of the project that stores the bucket.\n            region (str): The region of the bucket.\n\n        Returns:\n            A string representing the created bucket name.\n        "
        if self.bucket_name is None:
            self.bucket_name = '{}-automl-tables-staging'.format(project)
        try:
            self.client.get_bucket(self.bucket_name)
        except (exceptions.Forbidden, exceptions.NotFound) as e:
            if isinstance(e, exceptions.Forbidden):
                used_bucket_name = self.bucket_name
                self.bucket_name = used_bucket_name + '-{}'.format(int(time.time()))
                _LOGGER.warning('Created a bucket named {} because a bucket named {} already exists in a different project.'.format(self.bucket_name, used_bucket_name))
            bucket = self.client.bucket(self.bucket_name)
            bucket.create(project=project, location=region)
        return self.bucket_name

    def upload_pandas_dataframe(self, dataframe, uploaded_csv_name=None):
        if False:
            i = 10
            return i + 15
        'Uploads a Pandas DataFrame as CSV to the bucket.\n\n        Args:\n            dataframe (pandas.DataFrame): The Pandas Dataframe to be uploaded.\n            uploaded_csv_name (Optional[str]): The name for the uploaded CSV.\n\n        Returns:\n            A string representing the GCS URI of the uploaded CSV.\n        '
        if pandas is None:
            raise ImportError(_PANDAS_REQUIRED)
        if not isinstance(dataframe, pandas.DataFrame):
            raise ValueError("'dataframe' must be a pandas.DataFrame instance.")
        if self.bucket_name is None:
            raise ValueError('Must ensure a bucket exists before uploading data.')
        if uploaded_csv_name is None:
            uploaded_csv_name = 'automl-tables-dataframe-{}.csv'.format(int(time.time()))
        csv_string = dataframe.to_csv(index=False)
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(uploaded_csv_name)
        blob.upload_from_string(csv_string)
        return 'gs://{}/{}'.format(self.bucket_name, uploaded_csv_name)