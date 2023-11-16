from os import path
import unittest
from ..util import LanghostTest

class RemoteComponentProvidersTest(LanghostTest):

    def test_remote_component_dependencies(self):
        if False:
            return 10
        self.run_test(program=path.join(self.base_path(), 'remote_component_providers'), expected_resource_count=8)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, _protect, provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, providers, source_position):
        if False:
            while True:
                i = 10
        if name == 'singular' or name == 'map' or name == 'array':
            self.assertEqual(provider, 'myprovider::myprovider')
            self.assertEqual(list(providers.keys()), ['test'])
        if name == 'foo-singular' or name == 'foo-map' or name == 'foo-array':
            self.assertEqual(provider, '')
            self.assertEqual(list(providers.keys()), ['foo'])
        return {'urn': name, 'id': name}