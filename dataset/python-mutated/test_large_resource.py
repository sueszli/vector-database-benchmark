from os import path
import unittest
from ..util import LanghostTest
long_string = 'a' * 1024 * 1024 * 5

class LargeResourceTest(LanghostTest):

    def test_large_resource(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test(program=path.join(self.base_path(), 'large_resource'), expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            i = 10
            return i + 15
        self.assertEqual(ty, 'test:index:MyLargeStringResource')
        self.assertEqual(name, 'testResource1')
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': {'largeStringProp': long_string}}