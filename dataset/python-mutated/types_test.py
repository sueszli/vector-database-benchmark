"""Unit tests for types module."""
import datetime
import logging
import unittest
import mock
try:
    from google.cloud.datastore import client
    from google.cloud.datastore import entity
    from google.cloud.datastore import key
    from google.cloud.datastore.helpers import GeoPoint
    from apache_beam.io.gcp.datastore.v1new.types import Entity
    from apache_beam.io.gcp.datastore.v1new.types import Key
    from apache_beam.io.gcp.datastore.v1new.types import Query
    from apache_beam.options.value_provider import StaticValueProvider
except ImportError:
    client = None
_LOGGER = logging.getLogger(__name__)

@unittest.skipIf(client is None, 'Datastore dependencies are not installed')
class TypesTest(unittest.TestCase):
    _PROJECT = 'project'
    _NAMESPACE = 'namespace'

    def setUp(self):
        if False:
            while True:
                i = 10
        self._test_client = client.Client(project=self._PROJECT, namespace=self._NAMESPACE, _http=mock.MagicMock())

    def _assert_keys_equal(self, beam_type, client_type, expected_project):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(beam_type.path_elements[0], client_type.kind)
        self.assertEqual(beam_type.path_elements[1], client_type.id)
        self.assertEqual(expected_project, client_type.project)

    def testEntityToClientEntity(self):
        if False:
            i = 10
            return i + 15
        k = Key(['kind', 1234], project=self._PROJECT)
        kc = k.to_client_key()
        exclude_from_indexes = ('datetime', 'key')
        e = Entity(k, exclude_from_indexes=exclude_from_indexes)
        properties = {'datetime': datetime.datetime.utcnow(), 'key_ref': Key(['kind2', 1235]), 'bool': True, 'float': 1.21, 'int': 1337, 'unicode': 'text', 'bytes': b'bytes', 'geopoint': GeoPoint(0.123, 0.456), 'none': None, 'list': [1, 2, 3], 'entity': Entity(Key(['kind', 111])), 'dict': {'property': 5}}
        e.set_properties(properties)
        ec = e.to_client_entity()
        self.assertEqual(kc, ec.key)
        self.assertSetEqual(set(exclude_from_indexes), ec.exclude_from_indexes)
        self.assertEqual('kind', ec.kind)
        self.assertEqual(1234, ec.id)
        for (name, unconverted) in properties.items():
            converted = ec[name]
            if name == 'key_ref':
                self.assertNotIsInstance(converted, Key)
                self._assert_keys_equal(unconverted, converted, self._PROJECT)
            elif name == 'entity':
                self.assertNotIsInstance(converted, Entity)
                self.assertNotIsInstance(converted.key, Key)
                self._assert_keys_equal(unconverted.key, converted.key, self._PROJECT)
            else:
                self.assertEqual(unconverted, converted)
        entity_from_client_entity = Entity.from_client_entity(ec)
        self.assertEqual(e, entity_from_client_entity)

    def testEmbeddedClientEntityWithoutKey(self):
        if False:
            i = 10
            return i + 15
        client_entity = entity.Entity(key.Key('foo', project='bar'))
        entity_without_key = entity.Entity()
        entity_without_key['test'] = True
        client_entity['embedded'] = entity_without_key
        e = Entity.from_client_entity(client_entity)
        self.assertIsInstance(e.properties['embedded'], dict)

    def testKeyToClientKey(self):
        if False:
            while True:
                i = 10
        k = Key(['kind1', 'parent'], project=self._PROJECT, namespace=self._NAMESPACE)
        ck = k.to_client_key()
        self.assertEqual(self._PROJECT, ck.project)
        self.assertEqual(self._NAMESPACE, ck.namespace)
        self.assertEqual(('kind1', 'parent'), ck.flat_path)
        self.assertEqual('kind1', ck.kind)
        self.assertEqual('parent', ck.id_or_name)
        self.assertEqual(None, ck.parent)
        k2 = Key(['kind2', 1234], parent=k)
        ck2 = k2.to_client_key()
        self.assertEqual(self._PROJECT, ck2.project)
        self.assertEqual(self._NAMESPACE, ck2.namespace)
        self.assertEqual(('kind1', 'parent', 'kind2', 1234), ck2.flat_path)
        self.assertEqual('kind2', ck2.kind)
        self.assertEqual(1234, ck2.id_or_name)
        self.assertEqual(ck, ck2.parent)

    def testKeyFromClientKey(self):
        if False:
            for i in range(10):
                print('nop')
        k = Key(['k1', 1234], project=self._PROJECT, namespace=self._NAMESPACE)
        kfc = Key.from_client_key(k.to_client_key())
        self.assertEqual(k, kfc)
        k2 = Key(['k2', 'adsf'], parent=k)
        kfc2 = Key.from_client_key(k2.to_client_key())
        self.assertNotEqual(k2, kfc2)
        self.assertTupleEqual(('k1', 1234, 'k2', 'adsf'), kfc2.path_elements)
        self.assertIsNone(kfc2.parent)
        kfc3 = Key.from_client_key(kfc2.to_client_key())
        self.assertEqual(kfc2, kfc3)
        kfc4 = Key.from_client_key(kfc2.to_client_key())
        kfc4.project = 'other'
        self.assertNotEqual(kfc2, kfc4)

    def testKeyFromClientKeyNoNamespace(self):
        if False:
            for i in range(10):
                print('nop')
        k = Key(['k1', 1234], project=self._PROJECT)
        ck = k.to_client_key()
        self.assertEqual(None, ck.namespace)
        kfc = Key.from_client_key(ck)
        self.assertEqual(k, kfc)

    def testKeyToClientKeyMissingProject(self):
        if False:
            print('Hello World!')
        k = Key(['k1', 1234], namespace=self._NAMESPACE)
        with self.assertRaisesRegex(ValueError, 'project'):
            _ = Key.from_client_key(k.to_client_key())

    def testQuery(self):
        if False:
            return 10
        filters = [('property_name', '=', 'value')]
        projection = ['f1', 'f2']
        order = projection
        distinct_on = projection
        ancestor_key = Key(['kind', 'id'], project=self._PROJECT)
        q = Query(kind='kind', project=self._PROJECT, namespace=self._NAMESPACE, ancestor=ancestor_key, filters=filters, projection=projection, order=order, distinct_on=distinct_on)
        cq = q._to_client_query(self._test_client)
        self.assertEqual(self._PROJECT, cq.project)
        self.assertEqual(self._NAMESPACE, cq.namespace)
        self.assertEqual('kind', cq.kind)
        self.assertEqual(ancestor_key.to_client_key(), cq.ancestor)
        self.assertEqual(filters, cq.filters)
        self.assertEqual(projection, cq.projection)
        self.assertEqual(order, cq.order)
        self.assertEqual(distinct_on, cq.distinct_on)
        _LOGGER.info('query: %s', q)

    def testValueProviderFilters(self):
        if False:
            while True:
                i = 10
        self.vp_filters = [[(StaticValueProvider(str, 'property_name'), StaticValueProvider(str, '='), StaticValueProvider(str, 'value'))], [(StaticValueProvider(str, 'property_name'), StaticValueProvider(str, '='), StaticValueProvider(str, 'value')), ('property_name', '=', 'value')]]
        self.expected_filters = [[('property_name', '=', 'value')], [('property_name', '=', 'value'), ('property_name', '=', 'value')]]
        for (vp_filter, exp_filter) in zip(self.vp_filters, self.expected_filters):
            q = Query(kind='kind', project=self._PROJECT, namespace=self._NAMESPACE, filters=vp_filter)
            cq = q._to_client_query(self._test_client)
            self.assertEqual(exp_filter, cq.filters)
            _LOGGER.info('query: %s', q)

    def testValueProviderNamespace(self):
        if False:
            while True:
                i = 10
        self.vp_namespace = StaticValueProvider(str, 'vp_namespace')
        self.expected_namespace = 'vp_namespace'
        q = Query(kind='kind', project=self._PROJECT, namespace=self.vp_namespace)
        cq = q._to_client_query(self._test_client)
        self.assertEqual(self.expected_namespace, cq.namespace)
        _LOGGER.info('query: %s', q)

    def testQueryEmptyNamespace(self):
        if False:
            print('Hello World!')
        self._test_client.namespace = None
        q = Query(project=self._PROJECT, namespace=None)
        cq = q._to_client_query(self._test_client)
        self.assertEqual(self._test_client.project, cq.project)
        self.assertEqual(None, cq.namespace)
if __name__ == '__main__':
    unittest.main()