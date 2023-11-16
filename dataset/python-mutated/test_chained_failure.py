from os import path
from ..util import LanghostTest

class ChainedFailureTest(LanghostTest):
    """
    Tests that the language host can tolerate "chained failures" - that is, a failure of an output to resolve when
    attempting to prepare a resource for registration.

    In this test, the program raises an exception in an apply, which causes the preparation of resourceB to fail. This
    test asserts that this does not cause a deadlock (as it previously did, pulumi/pulumi#2189) but instead terminates
    gracefully.
    """

    def test_chained_failure(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'chained_failure'), expected_bail=True, expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            for i in range(10):
                print('nop')
        if ty == 'test:index:ResourceA':
            self.assertEqual(name, 'resourceA')
            self.assertDictEqual(_resource, {'inprop': 777})
            return {'urn': self.make_urn(ty, name), 'id': name, 'object': {'outprop': 200}}
        if ty == 'test:index:ResourceB':
            self.fail(f'we should never have gotten here! {_resource}')
        self.fail(f'unknown resource type: {ty}')