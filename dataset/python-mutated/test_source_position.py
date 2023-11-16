from os import path
from ..util import LanghostTest

class SourcePositionTest(LanghostTest):

    def test_source_position(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'source_position'), expected_resource_count=2)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            i = 10
            return i + 15
        assert source_position is not None
        assert source_position.uri.endswith('__main__.py')
        if name == 'custom':
            self.assertEqual(source_position.line, 24)
        elif name == 'component':
            self.assertEqual(source_position.line, 25)
        else:
            assert False
        return {'urn': self.make_urn(ty, name)}