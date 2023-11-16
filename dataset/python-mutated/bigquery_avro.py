"""Specialized tasks for handling Avro data in BigQuery from GCS.
"""
import logging
from luigi.contrib.bigquery import BigQueryLoadTask, SourceFormat
from luigi.contrib.gcs import GCSClient
from luigi.task import flatten
logger = logging.getLogger('luigi-interface')
try:
    import avro
    import avro.datafile
except ImportError:
    logger.warning('bigquery_avro module imported, but avro is not installed. Any BigQueryLoadAvro task will fail to propagate schema documentation')

class BigQueryLoadAvro(BigQueryLoadTask):
    """A helper for loading specifically Avro data into BigQuery from GCS.

    Copies table level description from Avro schema doc,
    BigQuery internally will copy field-level descriptions to the table.

    Suitable for use via subclassing: override requires() to return Task(s) that output
    to GCS Targets; their paths are expected to be URIs of .avro files or URI prefixes
    (GCS "directories") containing one or many .avro files.

    Override output() to return a BigQueryTarget representing the destination table.
    """
    source_format = SourceFormat.AVRO

    def _avro_uri(self, target):
        if False:
            return 10
        path_or_uri = target.uri if hasattr(target, 'uri') else target.path
        return path_or_uri if path_or_uri.endswith('.avro') else path_or_uri.rstrip('/') + '/*.avro'

    def source_uris(self):
        if False:
            return 10
        return [self._avro_uri(x) for x in flatten(self.input())]

    def _get_input_schema(self):
        if False:
            for i in range(10):
                print('nop')
        'Arbitrarily picks an object in input and reads the Avro schema from it.'
        assert avro, 'avro module required'
        input_target = flatten(self.input())[0]
        input_fs = input_target.fs if hasattr(input_target, 'fs') else GCSClient()
        input_uri = self.source_uris()[0]
        if '*' in input_uri:
            file_uris = list(input_fs.list_wildcard(input_uri))
            if file_uris:
                input_uri = file_uris[0]
            else:
                raise RuntimeError('No match for ' + input_uri)
        schema = []
        exception_reading_schema = []

        def read_schema(fp):
            if False:
                print('Hello World!')
            try:
                reader = avro.datafile.DataFileReader(fp, avro.io.DatumReader())
                schema[:] = [BigQueryLoadAvro._get_writer_schema(reader.datum_reader)]
            except Exception as e:
                exception_reading_schema[:] = [e]
                return False
            return True
        input_fs.download(input_uri, 64 * 1024, read_schema).close()
        if not schema:
            raise exception_reading_schema[0]
        return schema[0]

    @staticmethod
    def _get_writer_schema(datum_reader):
        if False:
            while True:
                i = 10
        'Python-version agnostic getter for datum_reader writer(s)_schema attribute\n\n        Parameters:\n        datum_reader (avro.io.DatumReader): DatumReader\n\n        Returns:\n        Returning correct attribute name depending on Python version.\n        '
        return datum_reader.writer_schema

    def _set_output_doc(self, avro_schema):
        if False:
            print('Hello World!')
        bq_client = self.output().client.client
        table = self.output().table
        patch = {'description': avro_schema.doc}
        bq_client.tables().patch(projectId=table.project_id, datasetId=table.dataset_id, tableId=table.table_id, body=patch).execute()

    def run(self):
        if False:
            return 10
        super(BigQueryLoadAvro, self).run()
        try:
            self._set_output_doc(self._get_input_schema())
        except Exception as e:
            logger.warning('Could not propagate Avro doc to BigQuery table description: %r', e)