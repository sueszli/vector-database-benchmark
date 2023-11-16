from os import path
from ..util import LanghostTest

class UnhandledExceptionTest(LanghostTest):

    def test_unhandled_exception(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'resource_op_bad_inputs'), expected_log_message='unexpected input of type MyClass', expected_bail=True)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            i = 10
            return i + 15
        raise Exception('oh no')