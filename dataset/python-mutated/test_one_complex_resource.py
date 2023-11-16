from os import path
import unittest
from ..util import LanghostTest

class OneComplexResourceTest(LanghostTest):

    def test_one_complex_resource(self):
        if False:
            return 10
        self.run_test(program=path.join(self.base_path(), 'one_complex_resource'), expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            i = 10
            return i + 15
        self.assertEqual(ty, 'test:index:MyResource')
        self.assertEqual(name, 'testres')
        self.assertEqual(_resource['falseprop'], False)
        self.assertEqual(_resource['trueprop'], True)
        self.assertEqual(_resource['intprop'], 42)
        self.assertListEqual(_resource['listprop'], [1, 2, 'string', False])
        self.assertDictEqual(_resource['mapprop'], {'foo': ['bar', 'baz']})
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': {'outprop': 'output properties ftw', 'outintprop': 99}}