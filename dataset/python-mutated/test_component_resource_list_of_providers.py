from os import path
from ..util import LanghostTest

class ComponentResourceListOfProvidersTest(LanghostTest):
    """
    Tests that resources inherit a variety of properties from their parents, when parents are present.

    This test generates a multi-level tree of resources of different kinds and parents them in various ways.
    The crux of the test is that all leaf resources in the resource tree should inherit a variety of properties
    from their parents.
    """

    def test_component_resource_list_of_providers(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'component_resource_list_of_providers'), expected_resource_count=240)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            while True:
                i = 10
        if _custom and (not ty.startswith('pulumi:providers:')):
            expect_protect = False
            expect_provider_name = ''
            rpath = name.split('/')
            for (i, component) in enumerate(rpath[1:]):
                if component in ['r0', 'c0']:
                    continue
                if component in ['r1', 'c1']:
                    expect_protect = False
                    continue
                if component in ['r2', 'c2']:
                    expect_protect = True
                    continue
                if component in ['r3', 'c3']:
                    expect_provider_name = '/'.join(rpath[:i + 1]) + '-p'
            if rpath[-1] == 'r3':
                expect_provider_name = '/'.join(rpath[:-1]) + '-p'
            provider_name = _provider.split('::')[-1]
            self.assertEqual(f'{name}.protect: {protect}', f'{name}.protect: {expect_protect}')
            self.assertEqual(f'{name}.provider: {provider_name}', f'{name}.provider: {expect_provider_name}')
        return {'urn': self.make_urn(ty, name), 'id': name}