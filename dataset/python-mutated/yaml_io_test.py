import io
import json
import logging
import unittest
import fastavro
import mock
import apache_beam as beam
from apache_beam.io.gcp.pubsub import PubsubMessage
from apache_beam.testing.util import AssertThat
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.yaml.yaml_transform import YamlTransform

class FakeReadFromPubSub:

    def __init__(self, topic, messages, subscription=None, id_attribute=None, timestamp_attribute=None):
        if False:
            i = 10
            return i + 15
        self._topic = topic
        self._subscription = subscription
        self._messages = messages
        self._id_attribute = id_attribute
        self._timestamp_attribute = timestamp_attribute

    def __call__(self, *, topic, subscription, with_attributes, id_label, timestamp_attribute):
        if False:
            print('Hello World!')
        assert topic == self._topic
        assert id_label == self._id_attribute
        assert timestamp_attribute == self._timestamp_attribute
        assert subscription == self._subscription
        if with_attributes:
            data = self._messages
        else:
            data = [x.data for x in self._messages]
        return beam.Create(data)

class FakeWriteToPubSub:

    def __init__(self, topic, messages, id_attribute=None, timestamp_attribute=None):
        if False:
            i = 10
            return i + 15
        self._topic = topic
        self._messages = messages
        self._id_attribute = id_attribute
        self._timestamp_attribute = timestamp_attribute

    def __call__(self, topic, *, with_attributes, id_label, timestamp_attribute):
        if False:
            return 10
        assert topic == self._topic
        assert with_attributes is True
        assert id_label == self._id_attribute
        assert timestamp_attribute == self._timestamp_attribute
        return AssertThat(equal_to(self._messages))

class YamlPubSubTest(unittest.TestCase):

    def test_simple_read(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.ReadFromPubSub', FakeReadFromPubSub(topic='my_topic', messages=[PubsubMessage(b'msg1', {'attr': 'value1'}), PubsubMessage(b'msg2', {'attr': 'value2'})])):
                result = p | YamlTransform('\n            type: ReadFromPubSub\n            config:\n              topic: my_topic\n              format: raw\n            ')
                assert_that(result, equal_to([beam.Row(payload=b'msg1'), beam.Row(payload=b'msg2')]))

    def test_read_with_attribute(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.ReadFromPubSub', FakeReadFromPubSub(topic='my_topic', messages=[PubsubMessage(b'msg1', {'attr': 'value1'}), PubsubMessage(b'msg2', {'attr': 'value2'})])):
                result = p | YamlTransform('\n            type: ReadFromPubSub\n            config:\n              topic: my_topic\n              format: raw\n              attributes: [attr]\n            ')
                assert_that(result, equal_to([beam.Row(payload=b'msg1', attr='value1'), beam.Row(payload=b'msg2', attr='value2')]))

    def test_read_with_attribute_map(self):
        if False:
            return 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.ReadFromPubSub', FakeReadFromPubSub(topic='my_topic', messages=[PubsubMessage(b'msg1', {'attr': 'value1'}), PubsubMessage(b'msg2', {'attr': 'value2'})])):
                result = p | YamlTransform('\n            type: ReadFromPubSub\n            config:\n              topic: my_topic\n              format: raw\n              attributes_map: attrMap\n            ')
                assert_that(result, equal_to([beam.Row(payload=b'msg1', attrMap={'attr': 'value1'}), beam.Row(payload=b'msg2', attrMap={'attr': 'value2'})]))

    def test_read_with_id_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.ReadFromPubSub', FakeReadFromPubSub(topic='my_topic', messages=[PubsubMessage(b'msg1', {'attr': 'value1'}), PubsubMessage(b'msg2', {'attr': 'value2'})], id_attribute='some_attr')):
                result = p | YamlTransform('\n            type: ReadFromPubSub\n            config:\n              topic: my_topic\n              format: raw\n              id_attribute: some_attr\n            ')
                assert_that(result, equal_to([beam.Row(payload=b'msg1'), beam.Row(payload=b'msg2')]))
    _avro_schema = {'type': 'record', 'name': 'ec', 'fields': [{'name': 'label', 'type': 'string'}, {'name': 'rank', 'type': 'int'}]}

    def _encode_avro(self, data):
        if False:
            while True:
                i = 10
        buffer = io.BytesIO()
        fastavro.schemaless_writer(buffer, self._avro_schema, data)
        buffer.seek(0)
        return buffer.read()

    def test_read_avro(self):
        if False:
            print('Hello World!')
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.ReadFromPubSub', FakeReadFromPubSub(topic='my_topic', messages=[PubsubMessage(self._encode_avro({'label': '37a', 'rank': 1}), {}), PubsubMessage(self._encode_avro({'label': '389a', 'rank': 2}), {})])):
                result = p | YamlTransform('\n            type: ReadFromPubSub\n            config:\n              topic: my_topic\n              format: avro\n              schema: %s\n            ' % json.dumps(self._avro_schema))
                assert_that(result, equal_to([beam.Row(label='37a', rank=1), beam.Row(label='389a', rank=2)]))

    def test_read_json(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.ReadFromPubSub', FakeReadFromPubSub(topic='my_topic', messages=[PubsubMessage(b'{"generator": {"x": 0, "y": 0}, "rank": 1}', {'weierstrass': 'y^2+y=x^3-x', 'label': '37a'})])):
                result = p | YamlTransform('\n            type: ReadFromPubSub\n            config:\n              topic: my_topic\n              format: json\n              schema:\n                type: object\n                properties:\n                  generator:\n                    type: object\n                    properties:\n                      x: {type: integer}\n                      y: {type: integer}\n                  rank: {type: integer}\n              attributes: [label]\n              attributes_map: other\n            ')
                assert_that(result, equal_to([beam.Row(generator=beam.Row(x=0, y=0), rank=1, label='37a', other={'label': '37a', 'weierstrass': 'y^2+y=x^3-x'})]))

    def test_read_json_with_error_handling(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.ReadFromPubSub', FakeReadFromPubSub(topic='my_topic', messages=[PubsubMessage('{"some_int": 123}', attributes={}), PubsubMessage('unparsable', attributes={})])):
                result = p | YamlTransform('\n            type: ReadFromPubSub\n            config:\n              topic: my_topic\n              format: json\n              schema:\n                type: object\n                properties:\n                  some_int: {type: integer}\n              error_handling:\n                output: errors\n            ')
                assert_that(result['good'], equal_to([beam.Row(some_int=123)]), label='CheckGood')
                assert_that(result['errors'] | beam.Map(lambda error: error.element), equal_to(['unparsable']), label='CheckErrors')

    def test_read_json_without_error_handling(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(Exception):
            with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
                with mock.patch('apache_beam.io.ReadFromPubSub', FakeReadFromPubSub(topic='my_topic', messages=[PubsubMessage('{"some_int": 123}', attributes={}), PubsubMessage('unparsable', attributes={})])):
                    _ = p | YamlTransform('\n              type: ReadFromPubSub\n              config:\n                topic: my_topic\n                format: json\n                schema:\n                  type: object\n                  properties:\n                    some_int: {type: integer}\n              ')

    def test_read_json_with_bad_schema(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.ReadFromPubSub', FakeReadFromPubSub(topic='my_topic', messages=[PubsubMessage('{"some_int": 123}', attributes={}), PubsubMessage('{"some_int": "NOT"}', attributes={})])):
                result = p | YamlTransform('\n            type: ReadFromPubSub\n            config:\n              topic: my_topic\n              format: json\n              schema:\n                type: object\n                properties:\n                  some_int: {type: integer}\n              error_handling:\n                output: errors\n            ')
                assert_that(result['good'], equal_to([beam.Row(some_int=123)]), label='CheckGood')
                assert_that(result['errors'] | beam.Map(lambda error: error.element), equal_to(['{"some_int": "NOT"}']), label='CheckErrors')

    def test_simple_write(self):
        if False:
            return 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.WriteToPubSub', FakeWriteToPubSub(topic='my_topic', messages=[PubsubMessage(b'msg1', {}), PubsubMessage(b'msg2', {})])):
                _ = p | beam.Create([beam.Row(a=b'msg1'), beam.Row(a=b'msg2')]) | YamlTransform('\n            type: WriteToPubSub\n            config:\n              topic: my_topic\n              format: raw\n            ')

    def test_write_with_attribute(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.WriteToPubSub', FakeWriteToPubSub(topic='my_topic', messages=[PubsubMessage(b'msg1', {'attr': 'value1'}), PubsubMessage(b'msg2', {'attr': 'value2'})])):
                _ = p | beam.Create([beam.Row(a=b'msg1', attr='value1'), beam.Row(a=b'msg2', attr='value2')]) | YamlTransform('\n            type: WriteToPubSub\n            config:\n              topic: my_topic\n              format: raw\n              attributes: [attr]\n            ')

    def test_write_with_attribute_map(self):
        if False:
            return 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.WriteToPubSub', FakeWriteToPubSub(topic='my_topic', messages=[PubsubMessage(b'msg1', {'a': 'b'}), PubsubMessage(b'msg2', {'c': 'd'})])):
                _ = p | beam.Create([beam.Row(a=b'msg1', attrMap={'a': 'b'}), beam.Row(a=b'msg2', attrMap={'c': 'd'})]) | YamlTransform('\n            type: WriteToPubSub\n            config:\n              topic: my_topic\n              format: raw\n              attributes_map: attrMap\n            ')

    def test_write_with_id_attribute(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.WriteToPubSub', FakeWriteToPubSub(topic='my_topic', messages=[PubsubMessage(b'msg1', {}), PubsubMessage(b'msg2', {})], id_attribute='some_attr')):
                _ = p | beam.Create([beam.Row(a=b'msg1'), beam.Row(a=b'msg2')]) | YamlTransform('\n            type: WriteToPubSub\n            config:\n              topic: my_topic\n              format: raw\n              id_attribute: some_attr\n            ')

    def test_write_avro(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.WriteToPubSub', FakeWriteToPubSub(topic='my_topic', messages=[PubsubMessage(self._encode_avro({'label': '37a', 'rank': 1}), {}), PubsubMessage(self._encode_avro({'label': '389a', 'rank': 2}), {})])):
                _ = p | beam.Create([beam.Row(label='37a', rank=1), beam.Row(label='389a', rank=2)]) | YamlTransform('\n            type: WriteToPubSub\n            config:\n              topic: my_topic\n              format: avro\n            ')

    def test_write_json(self):
        if False:
            return 10
        with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions(pickle_library='cloudpickle')) as p:
            with mock.patch('apache_beam.io.WriteToPubSub', FakeWriteToPubSub(topic='my_topic', messages=[PubsubMessage(b'{"generator": {"x": 0, "y": 0}, "rank": 1}', {'weierstrass': 'y^2+y=x^3-x', 'label': '37a'})])):
                _ = p | beam.Create([beam.Row(label='37a', generator=beam.Row(x=0, y=0), rank=1, other={'weierstrass': 'y^2+y=x^3-x'})]) | YamlTransform('\n            type: WriteToPubSub\n            config:\n              topic: my_topic\n              format: json\n              attributes: [label]\n              attributes_map: other\n            ')
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()