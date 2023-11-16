import json
from os import path
from ..util import LanghostTest

class InputTypeMismatchTest(LanghostTest):

    def test_input_type_mismatch(self):
        if False:
            return 10
        self.run_test(program=path.join(self.base_path(), 'input_type_mismatch'), expected_resource_count=2)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            while True:
                i = 10
        self.assertEqual('test:index:MyResource', ty)
        policy = _resource['policy']
        if isinstance(policy, dict):
            policy = json.dumps(policy)
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': {'policy': policy}}

    def register_resource_outputs(self, _ctx, _dry_run, _urn, ty, _name, _resource, outputs):
        if False:
            i = 10
            return i + 15
        self.assertEqual('pulumi:pulumi:Stack', ty)
        self.assertEqual({'r1.policy': '{"hello": "world"}', 'r2.policy': '{"hello": "world"}'}, outputs)