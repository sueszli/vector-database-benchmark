"""BigTable connector

This module implements writing to BigTable tables.
The default mode is to set row data to write to BigTable tables.
The syntax supported is described here:
https://cloud.google.com/bigtable/docs/quickstart-cbt

BigTable connector can be used as main outputs. A main output
(common case) is expected to be massive and will be split into
manageable chunks and processed in parallel. In the example below
we created a list of rows then passed to the GeneratedDirectRows
DoFn to set the Cells and then we call the BigTableWriteFn to insert
those generated rows in the table.

  main_table = (p
                | beam.Create(self._generate())
                | WriteToBigTable(project_id,
                                  instance_id,
                                  table_id))
"""
import logging
import struct
from typing import Dict
from typing import List
import apache_beam as beam
from apache_beam.internal.metrics.metric import ServiceCallMetric
from apache_beam.io.gcp import resource_identifiers
from apache_beam.metrics import Metrics
from apache_beam.metrics import monitoring_infos
from apache_beam.transforms import PTransform
from apache_beam.transforms.display import DisplayDataItem
from apache_beam.transforms.external import BeamJarExpansionService
from apache_beam.transforms.external import SchemaAwareExternalTransform
from apache_beam.typehints.row_type import RowTypeConstraint
_LOGGER = logging.getLogger(__name__)
try:
    from google.cloud.bigtable import Client
    from google.cloud.bigtable.row import Cell, PartialRowData
    from google.cloud.bigtable.batcher import MutationsBatcher
    FLUSH_COUNT = 1000
    MAX_ROW_BYTES = 5242880
except ImportError:
    _LOGGER.warning('ImportError: from google.cloud.bigtable import Client', exc_info=True)
__all__ = ['WriteToBigTable', 'ReadFromBigtable']

class _BigTableWriteFn(beam.DoFn):
    """ Creates the connector can call and add_row to the batcher using each
  row in beam pipe line
  Args:
    project_id(str): GCP Project ID
    instance_id(str): GCP Instance ID
    table_id(str): GCP Table ID

  """

    def __init__(self, project_id, instance_id, table_id):
        if False:
            print('Hello World!')
        ' Constructor of the Write connector of Bigtable\n    Args:\n      project_id(str): GCP Project of to write the Rows\n      instance_id(str): GCP Instance to write the Rows\n      table_id(str): GCP Table to write the `DirectRows`\n    '
        super().__init__()
        self.beam_options = {'project_id': project_id, 'instance_id': instance_id, 'table_id': table_id}
        self.table = None
        self.batcher = None
        self.service_call_metric = None
        self.written = Metrics.counter(self.__class__, 'Written Row')

    def __getstate__(self):
        if False:
            print('Hello World!')
        return self.beam_options

    def __setstate__(self, options):
        if False:
            i = 10
            return i + 15
        self.beam_options = options
        self.table = None
        self.batcher = None
        self.service_call_metric = None
        self.written = Metrics.counter(self.__class__, 'Written Row')

    def write_mutate_metrics(self, status_list):
        if False:
            while True:
                i = 10
        for status in status_list:
            code = status.code if status else None
            grpc_status_string = ServiceCallMetric.bigtable_error_code_to_grpc_status_string(code)
            self.service_call_metric.call(grpc_status_string)

    def start_service_call_metrics(self, project_id, instance_id, table_id):
        if False:
            i = 10
            return i + 15
        resource = resource_identifiers.BigtableTable(project_id, instance_id, table_id)
        labels = {monitoring_infos.SERVICE_LABEL: 'BigTable', monitoring_infos.METHOD_LABEL: 'google.bigtable.v2.MutateRows', monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.BIGTABLE_PROJECT_ID_LABEL: self.beam_options['project_id'], monitoring_infos.INSTANCE_ID_LABEL: self.beam_options['instance_id'], monitoring_infos.TABLE_ID_LABEL: self.beam_options['table_id']}
        return ServiceCallMetric(request_count_urn=monitoring_infos.API_REQUEST_COUNT_URN, base_labels=labels)

    def start_bundle(self):
        if False:
            i = 10
            return i + 15
        if self.table is None:
            client = Client(project=self.beam_options['project_id'])
            instance = client.instance(self.beam_options['instance_id'])
            self.table = instance.table(self.beam_options['table_id'])
        self.service_call_metric = self.start_service_call_metrics(self.beam_options['project_id'], self.beam_options['instance_id'], self.beam_options['table_id'])
        self.batcher = MutationsBatcher(self.table, batch_completed_callback=self.write_mutate_metrics)

    def process(self, row):
        if False:
            for i in range(10):
                print('nop')
        self.written.inc()
        self.batcher.mutate(row)

    def finish_bundle(self):
        if False:
            for i in range(10):
                print('nop')
        if self.batcher:
            self.batcher.close()
            self.batcher = None

    def display_data(self):
        if False:
            print('Hello World!')
        return {'projectId': DisplayDataItem(self.beam_options['project_id'], label='Bigtable Project Id'), 'instanceId': DisplayDataItem(self.beam_options['instance_id'], label='Bigtable Instance Id'), 'tableId': DisplayDataItem(self.beam_options['table_id'], label='Bigtable Table Id')}

class WriteToBigTable(beam.PTransform):
    """A transform that writes rows to a Bigtable table.

  Takes an input PCollection of `DirectRow` objects containing un-committed
  mutations. For more information about this row object, visit
  https://cloud.google.com/python/docs/reference/bigtable/latest/row#class-googlecloudbigtablerowdirectrowrowkey-tablenone

  If flag `use_cross_language` is set to true, this transform will use the
  multi-language transforms framework to inject the Java native write transform
  into the pipeline.
  """
    URN = 'beam:schematransform:org.apache.beam:bigtable_write:v1'

    def __init__(self, project_id, instance_id, table_id, use_cross_language=False, expansion_service=None):
        if False:
            i = 10
            return i + 15
        'Initialize an WriteToBigTable transform.\n\n    :param table_id:\n      The ID of the table to write to.\n    :param instance_id:\n      The ID of the instance where the table resides.\n    :param project_id:\n      The GCP project ID.\n    :param use_cross_language:\n      If set to True, will use the Java native transform via cross-language.\n    :param expansion_service:\n      The address of the expansion service in the case of using cross-language.\n      If no expansion service is provided, will attempt to run the default GCP\n      expansion service.\n    '
        super().__init__()
        self._table_id = table_id
        self._instance_id = instance_id
        self._project_id = project_id
        self._use_cross_language = use_cross_language
        if use_cross_language:
            self._expansion_service = expansion_service or BeamJarExpansionService('sdks:java:io:google-cloud-platform:expansion-service:build')
            self.schematransform_config = SchemaAwareExternalTransform.discover_config(self._expansion_service, self.URN)

    def expand(self, input):
        if False:
            while True:
                i = 10
        if self._use_cross_language:
            external_write = SchemaAwareExternalTransform(identifier=self.schematransform_config.identifier, expansion_service=self._expansion_service, rearrange_based_on_discovery=True, tableId=self._table_id, instanceId=self._instance_id, projectId=self._project_id)
            return input | beam.ParDo(self._DirectRowMutationsToBeamRow()).with_output_types(RowTypeConstraint.from_fields([('key', bytes), ('mutations', List[Dict[str, bytes]])])) | external_write
        else:
            return input | beam.ParDo(_BigTableWriteFn(self._project_id, self._instance_id, self._table_id))

    class _DirectRowMutationsToBeamRow(beam.DoFn):

        def process(self, direct_row):
            if False:
                i = 10
                return i + 15
            args = {'key': direct_row.row_key, 'mutations': []}
            for mutation in direct_row._get_mutations():
                if mutation.__contains__('set_cell'):
                    mutation_dict = {'type': b'SetCell', 'family_name': mutation.set_cell.family_name.encode('utf-8'), 'column_qualifier': mutation.set_cell.column_qualifier, 'value': mutation.set_cell.value, 'timestamp_micros': struct.pack('>q', mutation.set_cell.timestamp_micros)}
                elif mutation.__contains__('delete_from_column'):
                    mutation_dict = {'type': b'DeleteFromColumn', 'family_name': mutation.delete_from_column.family_name.encode('utf-8'), 'column_qualifier': mutation.delete_from_column.column_qualifier}
                    time_range = mutation.delete_from_column.time_range
                    if time_range.start_timestamp_micros:
                        mutation_dict['start_timestamp_micros'] = struct.pack('>q', time_range.start_timestamp_micros)
                    if time_range.end_timestamp_micros:
                        mutation_dict['end_timestamp_micros'] = struct.pack('>q', time_range.end_timestamp_micros)
                elif mutation.__contains__('delete_from_family'):
                    mutation_dict = {'type': b'DeleteFromFamily', 'family_name': mutation.delete_from_family.family_name.encode('utf-8')}
                elif mutation.__contains__('delete_from_row'):
                    mutation_dict = {'type': b'DeleteFromRow'}
                else:
                    raise ValueError('Unexpected mutation')
                args['mutations'].append(mutation_dict)
            yield beam.Row(**args)

class ReadFromBigtable(PTransform):
    """Reads rows from Bigtable.

  Returns a PCollection of PartialRowData objects, each representing a
  Bigtable row. For more information about this row object, visit
  https://cloud.google.com/python/docs/reference/bigtable/latest/row#class-googlecloudbigtablerowpartialrowdatarowkey
  """
    URN = 'beam:schematransform:org.apache.beam:bigtable_read:v1'

    def __init__(self, project_id, instance_id, table_id, expansion_service=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a ReadFromBigtable transform.\n\n    :param table_id:\n      The ID of the table to read from.\n    :param instance_id:\n      The ID of the instance where the table resides.\n    :param project_id:\n      The GCP project ID.\n    :param expansion_service:\n      The address of the expansion service. If no expansion service is\n      provided, will attempt to run the default GCP expansion service.\n    '
        super().__init__()
        self._table_id = table_id
        self._instance_id = instance_id
        self._project_id = project_id
        self._expansion_service = expansion_service or BeamJarExpansionService('sdks:java:io:google-cloud-platform:expansion-service:build')
        self.schematransform_config = SchemaAwareExternalTransform.discover_config(self._expansion_service, self.URN)

    def expand(self, input):
        if False:
            return 10
        external_read = SchemaAwareExternalTransform(identifier=self.schematransform_config.identifier, expansion_service=self._expansion_service, rearrange_based_on_discovery=True, tableId=self._table_id, instanceId=self._instance_id, projectId=self._project_id)
        return input.pipeline | external_read | beam.ParDo(self._BeamRowToPartialRowData())

    class _BeamRowToPartialRowData(beam.DoFn):

        def process(self, row):
            if False:
                return 10
            key = row.key
            families = row.column_families
            partial_row: PartialRowData = PartialRowData(key)
            for (fam_name, col_fam) in families.items():
                if fam_name not in partial_row.cells:
                    partial_row.cells[fam_name] = {}
                for (col_qualifier, cells) in col_fam.items():
                    col_qualifier_bytes = col_qualifier.encode()
                    if col_qualifier not in partial_row.cells[fam_name]:
                        partial_row.cells[fam_name][col_qualifier_bytes] = []
                    for cell in cells:
                        value = cell.value
                        timestamp_micros = cell.timestamp_micros
                        partial_row.cells[fam_name][col_qualifier_bytes].append(Cell(value, timestamp_micros))
            yield partial_row