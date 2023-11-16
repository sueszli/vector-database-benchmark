import argparse
import logging
import pickle
import signal
import sys
import typing
import grpc
import apache_beam as beam
import apache_beam.transforms.combiners as combine
from apache_beam.coders import RowCoder
from apache_beam.pipeline import PipelineOptions
from apache_beam.portability.api import beam_artifact_api_pb2_grpc
from apache_beam.portability.api import beam_expansion_api_pb2_grpc
from apache_beam.portability.api import external_transforms_pb2
from apache_beam.runners.portability import artifact_service
from apache_beam.runners.portability import expansion_service
from apache_beam.runners.portability.stager import Stager
from apache_beam.transforms import fully_qualified_named_transform
from apache_beam.transforms import ptransform
from apache_beam.transforms.environments import PyPIArtifactRegistry
from apache_beam.transforms.external import ImplicitSchemaPayloadBuilder
from apache_beam.utils import thread_pool_executor
_LOGGER = logging.getLogger(__name__)
TEST_PREFIX_URN = 'beam:transforms:xlang:test:prefix'
TEST_MULTI_URN = 'beam:transforms:xlang:test:multi'
TEST_GBK_URN = 'beam:transforms:xlang:test:gbk'
TEST_CGBK_URN = 'beam:transforms:xlang:test:cgbk'
TEST_COMGL_URN = 'beam:transforms:xlang:test:comgl'
TEST_COMPK_URN = 'beam:transforms:xlang:test:compk'
TEST_FLATTEN_URN = 'beam:transforms:xlang:test:flatten'
TEST_PARTITION_URN = 'beam:transforms:xlang:test:partition'
TEST_PYTHON_BS4_URN = 'beam:transforms:xlang:test:python_bs4'
TEST_NO_OUTPUT_URN = 'beam:transforms:xlang:test:nooutput'

@ptransform.PTransform.register_urn('beam:transforms:xlang:count', None)
class CountPerElementTransform(ptransform.PTransform):

    def expand(self, pcoll):
        if False:
            return 10
        return pcoll | combine.Count.PerElement()

    def to_runner_api_parameter(self, unused_context):
        if False:
            i = 10
            return i + 15
        return ('beam:transforms:xlang:count', None)

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, unused_parameter, unused_context):
        if False:
            for i in range(10):
                print('nop')
        return CountPerElementTransform()

@ptransform.PTransform.register_urn('beam:transforms:xlang:filter_less_than_eq', bytes)
class FilterLessThanTransform(ptransform.PTransform):

    def __init__(self, payload):
        if False:
            while True:
                i = 10
        self._payload = payload

    def expand(self, pcoll):
        if False:
            return 10
        return pcoll | beam.Filter(lambda elem, target: elem <= target, int(ord(self._payload[0])))

    def to_runner_api_parameter(self, unused_context):
        if False:
            print('Hello World!')
        return ('beam:transforms:xlang:filter_less_than', self._payload.encode('utf8'))

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, payload, unused_context):
        if False:
            for i in range(10):
                print('nop')
        return FilterLessThanTransform(payload.decode('utf8'))

@ptransform.PTransform.register_urn(TEST_PREFIX_URN, None)
@beam.typehints.with_output_types(str)
class PrefixTransform(ptransform.PTransform):

    def __init__(self, payload):
        if False:
            i = 10
            return i + 15
        self._payload = payload

    def expand(self, pcoll):
        if False:
            while True:
                i = 10
        return pcoll | 'TestLabel' >> beam.Map(lambda x: '{}{}'.format(self._payload, x))

    def to_runner_api_parameter(self, unused_context):
        if False:
            for i in range(10):
                print('nop')
        return (TEST_PREFIX_URN, ImplicitSchemaPayloadBuilder({'data': self._payload}).payload())

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, payload, unused_context):
        if False:
            while True:
                i = 10
        return PrefixTransform(parse_string_payload(payload)['data'])

@ptransform.PTransform.register_urn(TEST_MULTI_URN, None)
class MutltiTransform(ptransform.PTransform):

    def expand(self, pcolls):
        if False:
            print('Hello World!')
        return {'main': (pcolls['main1'], pcolls['main2']) | beam.Flatten() | beam.Map(lambda x, s: x + s, beam.pvalue.AsSingleton(pcolls['side'])).with_output_types(str), 'side': pcolls['side'] | beam.Map(lambda x: x + x).with_output_types(str)}

    def to_runner_api_parameter(self, unused_context):
        if False:
            i = 10
            return i + 15
        return (TEST_MULTI_URN, None)

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, unused_parameter, unused_context):
        if False:
            for i in range(10):
                print('nop')
        return MutltiTransform()

@ptransform.PTransform.register_urn(TEST_GBK_URN, None)
class GBKTransform(ptransform.PTransform):

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll | 'TestLabel' >> beam.GroupByKey()

    def to_runner_api_parameter(self, unused_context):
        if False:
            return 10
        return (TEST_GBK_URN, None)

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, unused_parameter, unused_context):
        if False:
            for i in range(10):
                print('nop')
        return GBKTransform()

@ptransform.PTransform.register_urn(TEST_CGBK_URN, None)
class CoGBKTransform(ptransform.PTransform):

    class ConcatFn(beam.DoFn):

        def process(self, element):
            if False:
                for i in range(10):
                    print('nop')
            (k, v) = element
            return [(k, v['col1'] + v['col2'])]

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll | beam.CoGroupByKey() | beam.ParDo(self.ConcatFn()).with_output_types(typing.Tuple[int, typing.Iterable[str]])

    def to_runner_api_parameter(self, unused_context):
        if False:
            return 10
        return (TEST_CGBK_URN, None)

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, unused_parameter, unused_context):
        if False:
            while True:
                i = 10
        return CoGBKTransform()

@ptransform.PTransform.register_urn(TEST_COMGL_URN, None)
class CombineGloballyTransform(ptransform.PTransform):

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll | beam.CombineGlobally(sum).with_output_types(int)

    def to_runner_api_parameter(self, unused_context):
        if False:
            for i in range(10):
                print('nop')
        return (TEST_COMGL_URN, None)

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, unused_parameter, unused_context):
        if False:
            i = 10
            return i + 15
        return CombineGloballyTransform()

@ptransform.PTransform.register_urn(TEST_COMPK_URN, None)
class CombinePerKeyTransform(ptransform.PTransform):

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        output = pcoll | beam.CombinePerKey(sum)
        output.element_type = beam.typehints.Tuple[str, int]
        return output

    def to_runner_api_parameter(self, unused_context):
        if False:
            i = 10
            return i + 15
        return (TEST_COMPK_URN, None)

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, unused_parameter, unused_context):
        if False:
            return 10
        return CombinePerKeyTransform()

@ptransform.PTransform.register_urn(TEST_FLATTEN_URN, None)
class FlattenTransform(ptransform.PTransform):

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll.values() | beam.Flatten().with_output_types(int)

    def to_runner_api_parameter(self, unused_context):
        if False:
            i = 10
            return i + 15
        return (TEST_FLATTEN_URN, None)

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, unused_parameter, unused_context):
        if False:
            i = 10
            return i + 15
        return FlattenTransform()

@ptransform.PTransform.register_urn(TEST_PARTITION_URN, None)
class PartitionTransform(ptransform.PTransform):

    def expand(self, pcoll):
        if False:
            while True:
                i = 10
        (col1, col2) = pcoll | beam.Partition(lambda elem, n: 0 if elem % 2 == 0 else 1, 2)
        typed_col1 = col1 | beam.Map(lambda x: x).with_output_types(int)
        typed_col2 = col2 | beam.Map(lambda x: x).with_output_types(int)
        return {'0': typed_col1, '1': typed_col2}

    def to_runner_api_parameter(self, unused_context):
        if False:
            while True:
                i = 10
        return (TEST_PARTITION_URN, None)

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, unused_parameter, unused_context):
        if False:
            i = 10
            return i + 15
        return PartitionTransform()

class ExtractHtmlTitleDoFn(beam.DoFn):

    def process(self, element):
        if False:
            for i in range(10):
                print('nop')
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(element, 'html.parser')
        return [soup.title.string]

@ptransform.PTransform.register_urn(TEST_PYTHON_BS4_URN, None)
class ExtractHtmlTitleTransform(ptransform.PTransform):

    def expand(self, pcoll):
        if False:
            return 10
        return pcoll | beam.ParDo(ExtractHtmlTitleDoFn()).with_output_types(str)

    def to_runner_api_parameter(self, unused_context):
        if False:
            return 10
        return (TEST_PYTHON_BS4_URN, None)

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, unused_parameter, unused_context):
        if False:
            for i in range(10):
                print('nop')
        return ExtractHtmlTitleTransform()

@ptransform.PTransform.register_urn('payload', bytes)
class PayloadTransform(ptransform.PTransform):

    def __init__(self, payload):
        if False:
            print('Hello World!')
        self._payload = payload

    def expand(self, pcoll):
        if False:
            return 10
        return pcoll | beam.Map(lambda x, s: x + s, self._payload)

    def to_runner_api_parameter(self, unused_context):
        if False:
            print('Hello World!')
        return (b'payload', self._payload.encode('ascii'))

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, payload, unused_context):
        if False:
            for i in range(10):
                print('nop')
        return PayloadTransform(payload.decode('ascii'))

@ptransform.PTransform.register_urn('map_to_union_types', None)
class MapToUnionTypesTransform(ptransform.PTransform):

    class CustomDoFn(beam.DoFn):

        def process(self, element):
            if False:
                i = 10
                return i + 15
            if element == 1:
                return ['1']
            elif element == 2:
                return [2]
            else:
                return [3.0]

    def expand(self, pcoll):
        if False:
            print('Hello World!')
        return pcoll | beam.ParDo(self.CustomDoFn())

    def to_runner_api_parameter(self, unused_context):
        if False:
            while True:
                i = 10
        return (b'map_to_union_types', None)

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, unused_payload, unused_context):
        if False:
            for i in range(10):
                print('nop')
        return MapToUnionTypesTransform()

@ptransform.PTransform.register_urn('fib', bytes)
class FibTransform(ptransform.PTransform):

    def __init__(self, level):
        if False:
            i = 10
            return i + 15
        self._level = level

    def expand(self, p):
        if False:
            return 10
        if self._level <= 2:
            return p | beam.Create([1])
        else:
            a = p | 'A' >> beam.ExternalTransform('fib', str(self._level - 1).encode('ascii'), expansion_service.ExpansionServiceServicer())
            b = p | 'B' >> beam.ExternalTransform('fib', str(self._level - 2).encode('ascii'), expansion_service.ExpansionServiceServicer())
            return (a, b) | beam.Flatten() | beam.CombineGlobally(sum).without_defaults()

    def to_runner_api_parameter(self, unused_context):
        if False:
            print('Hello World!')
        return ('fib', str(self._level).encode('ascii'))

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, level, unused_context):
        if False:
            print('Hello World!')
        return FibTransform(int(level.decode('ascii')))

@ptransform.PTransform.register_urn(TEST_NO_OUTPUT_URN, None)
class NoOutputTransform(ptransform.PTransform):

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15

        def log_val(val):
            if False:
                while True:
                    i = 10
            logging.debug('Got value: %r', val)
        _ = pcoll | 'TestLabel' >> beam.ParDo(log_val)

    def to_runner_api_parameter(self, unused_context):
        if False:
            for i in range(10):
                print('nop')
        return (TEST_NO_OUTPUT_URN, None)

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, payload, unused_context):
        if False:
            print('Hello World!')
        return NoOutputTransform(parse_string_payload(payload)['data'])

def parse_string_payload(input_byte):
    if False:
        i = 10
        return i + 15
    payload = external_transforms_pb2.ExternalConfigurationPayload()
    payload.ParseFromString(input_byte)
    return RowCoder(payload.schema).decode(payload.payload)._asdict()

def create_test_sklearn_model(file_name):
    if False:
        while True:
            i = 10
    from sklearn import svm
    x = [[0, 0], [1, 1]]
    y = [0, 1]
    model = svm.SVC()
    model.fit(x, y)
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def update_sklearn_model_dependency(env):
    if False:
        print('Hello World!')
    model_file = '/tmp/sklearn_test_model'
    staged_name = 'sklearn_model'
    create_test_sklearn_model(model_file)
    env._artifacts.append(Stager._create_file_stage_to_artifact(model_file, staged_name))
server = None

def cleanup(unused_signum, unused_frame):
    if False:
        i = 10
        return i + 15
    _LOGGER.info('Shutting down expansion service.')
    server.stop(None)

def main(unused_argv):
    if False:
        print('Hello World!')
    PyPIArtifactRegistry.register_artifact('beautifulsoup4', '>=4.9,<5.0')
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, help='port on which to serve the job api')
    parser.add_argument('--fully_qualified_name_glob', default=None)
    options = parser.parse_args()
    global server
    with fully_qualified_named_transform.FullyQualifiedNamedTransform.with_filter(options.fully_qualified_name_glob):
        server = grpc.server(thread_pool_executor.shared_unbounded_instance())
        expansion_servicer = expansion_service.ExpansionServiceServicer(PipelineOptions(['--experiments', 'beam_fn_api', '--sdk_location', 'container', '--pickle_library', 'cloudpickle']))
        update_sklearn_model_dependency(expansion_servicer._default_environment)
        beam_expansion_api_pb2_grpc.add_ExpansionServiceServicer_to_server(expansion_servicer, server)
        beam_artifact_api_pb2_grpc.add_ArtifactRetrievalServiceServicer_to_server(artifact_service.ArtifactRetrievalService(artifact_service.BeamFilesystemHandler(None).file_reader), server)
        server.add_insecure_port('localhost:{}'.format(options.port))
        server.start()
        _LOGGER.info('Listening for expansion requests at %d', options.port)
        signal.signal(signal.SIGTERM, cleanup)
        signal.signal(signal.SIGINT, cleanup)
        signal.pause()
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main(sys.argv)