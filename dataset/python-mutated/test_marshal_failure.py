from os import path
from ..util import LanghostTest

class TestMarshalFailure(LanghostTest):

    def test_marshal_failure(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test(program=path.join(self.base_path(), 'marshal_failure'), expected_resource_count=1, expected_bail=True)

    def invoke(self, _ctx, token, args, provider, _version):
        if False:
            return 10
        self.assertEqual('test:index:MyFunction', token)
        return ([], {})

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            print('Hello World!')
        self.assertEqual('test:index:MyResource', ty)
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': _resource}