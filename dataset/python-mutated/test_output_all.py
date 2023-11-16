from os import path
from ..util import LanghostTest

class OutputAllTest(LanghostTest):
    """
    """

    def test_output_all(self):
        if False:
            return 10
        self.run_test(program=path.join(self.base_path(), 'output_all'), expected_resource_count=4)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            while True:
                i = 10
        number = 0
        if name == 'testResource1':
            self.assertEqual(ty, 'test:index:MyResource')
            number = 2
        elif name == 'testResource2':
            self.assertEqual(ty, 'test:index:MyResource')
            number = 3
        elif name == 'testResource3':
            self.assertEqual(ty, 'test:index:FinalResource')
            self.assertEqual(_resource['number'], 5)
            number = _resource['number']
        elif name == 'testResource4':
            self.assertEqual(ty, 'test:index:FinalResource')
            self.assertEqual(_resource['number'], 5)
            number = _resource['number']
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': {'number': number}}