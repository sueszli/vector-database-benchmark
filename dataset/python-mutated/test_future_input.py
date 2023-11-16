from os import path
from ..util import LanghostTest

class FutureInputTest(LanghostTest):
    """
    Tests that Pulumi resources can accept awaitable objects as inputs
    to resources.
    """

    def test_future_input(self):
        if False:
            while True:
                i = 10
        self.run_test(program=path.join(self.base_path(), 'future_input'), expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            while True:
                i = 10
        self.assertEqual(ty, 'test:index:FileResource')
        self.assertEqual(name, 'file')
        self.assertDictEqual(_resource, {'contents': "here's a file"})
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': _resource}