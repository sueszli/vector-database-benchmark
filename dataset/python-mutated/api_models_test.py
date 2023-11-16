from collections.abc import Collection
import astroid
from astroid import parse, nodes
import pylint.checkers.typecheck
import pylint.testutils
try:
    from . import api_models
    FIXTURE_MODULE_ACTION = 'pylint_plugins.fixtures.api_models'
    FIXTURE_MODULE_TRIGGER = 'pylint_plugins.fixtures.api_models'
except ImportError:
    import api_models
    FIXTURE_MODULE_ACTION = 'fixtures.api_models'
    FIXTURE_MODULE_TRIGGER = 'fixtures.api_models'

def test_skiplist_class_gets_skipped():
    if False:
        i = 10
        return i + 15
    code = '\n    class ActionExecutionReRunController(object):\n        class ExecutionSpecificationAPI(object):\n            schema = {"properties": {"action": {}}}\n    '
    res = parse(code)
    assert isinstance(res, nodes.Module)
    assert isinstance(res.body, Collection)
    assert isinstance(res.body[0], nodes.ClassDef)
    class_node: nodes.ClassDef = res.body[0].body[0]
    assert isinstance(class_node, nodes.ClassDef)
    assert class_node.name in api_models.CLASS_NAME_SKIPLIST
    assert len(class_node.body) == 1
    assert 'schema' in class_node.locals
    assert 'action' not in class_node.locals
    assign_node: nodes.Assign = class_node.body[0]
    assert isinstance(assign_node, nodes.Assign)
    assert isinstance(assign_node.value, nodes.Dict)

def test_non_api_class_gets_skipped():
    if False:
        print('Hello World!')
    code = '\n    class ActionExecutionReRunController(object):\n        pass\n    '
    res = parse(code)
    assert isinstance(res, nodes.Module)
    class_node: nodes.ClassDef = res.body[0]
    assert isinstance(class_node, nodes.ClassDef)
    assert len(class_node.body) == 1
    assert isinstance(class_node.body[0], nodes.Pass)

def test_simple_schema():
    if False:
        return 10
    code = '\n    class ActionAPI(object):\n        schema = {"properties": {"action": {}}}\n    '
    res = parse(code)
    assert isinstance(res, nodes.Module)
    class_node: nodes.ClassDef = res.body[0]
    assert isinstance(class_node, nodes.ClassDef)
    assert 'schema' in class_node.locals
    assert 'action' in class_node.locals
    assert isinstance(class_node.locals['action'][0], nodes.AssignName)
    assert class_node.locals['action'][0].name == 'action'

def test_copied_schema():
    if False:
        for i in range(10):
            print('nop')
    code = '\n    import copy\n\n    class ActionAPI(object):\n        schema = {"properties": {"action": {}}}\n\n    class ActionCreateAPI(object):\n        schema = copy.deepcopy(ActionAPI.schema)\n        schema["properties"]["default_files"] = {}\n    '
    res = parse(code)
    assert isinstance(res, nodes.Module)
    class1_node: nodes.ClassDef = res.body[1]
    assert isinstance(class1_node, nodes.ClassDef)
    assert 'schema' in class1_node.locals
    assert 'action' in class1_node.locals
    assert 'default_files' not in class1_node.locals
    class2_node: nodes.ClassDef = res.body[2]
    assert isinstance(class2_node, nodes.ClassDef)
    assert 'schema' in class2_node.locals
    assert 'action' in class2_node.locals
    assert 'default_files' in class2_node.locals

def test_copied_imported_schema():
    if False:
        return 10
    code = '\n    import copy\n    from %s import ActionAPI\n\n    class ActionCreateAPI(object):\n        schema = copy.deepcopy(ActionAPI.schema)\n        schema["properties"]["default_files"] = {}\n    '
    code = code % FIXTURE_MODULE_ACTION
    res = parse(code)
    assert isinstance(res, nodes.Module)
    class_node: nodes.ClassDef = res.body[2]
    assert isinstance(class_node, nodes.ClassDef)
    assert 'schema' in class_node.locals
    assert 'name' in class_node.locals
    assert 'description' in class_node.locals
    assert 'runner_type' in class_node.locals
    assert 'default_files' in class_node.locals

def test_indirect_copied_schema():
    if False:
        i = 10
        return i + 15
    code = '\n    import copy\n    from %s import ActionAPI\n\n    REQUIRED_ATTR_SCHEMAS = {"action": copy.deepcopy(ActionAPI.schema)}\n\n    class ExecutionAPI(object):\n        schema = {"properties": {"action": REQUIRED_ATTR_SCHEMAS["action"]}}\n    '
    code = code % FIXTURE_MODULE_ACTION
    res = parse(code)
    assert isinstance(res, nodes.Module)
    class_node: nodes.ClassDef = res.body[3]
    assert isinstance(class_node, nodes.ClassDef)
    assert 'schema' in class_node.locals
    assert 'action' in class_node.locals
    attribute_value_node = next(class_node.locals['action'][0].infer())
    assert isinstance(attribute_value_node, nodes.Dict)

def test_inlined_schema():
    if False:
        print('Hello World!')
    code = '\n    from %s import TriggerAPI\n\n    class ActionExecutionAPI(object):\n        schema = {"properties": {"trigger": TriggerAPI.schema}}\n    '
    code = code % FIXTURE_MODULE_TRIGGER
    res = parse(code)
    assert isinstance(res, nodes.Module)
    class_node: nodes.ClassDef = res.body[1]
    assert isinstance(class_node, nodes.ClassDef)
    assert 'schema' in class_node.locals
    assert 'trigger' in class_node.locals
    attribute_value_node = next(class_node.locals['trigger'][0].infer())
    assert isinstance(attribute_value_node, nodes.Dict)

def test_property_types():
    if False:
        i = 10
        return i + 15
    code = '\n    class RandomAPI(object):\n        schema = {\n            "properties": {\n                "thing": {"type": "object"},\n                "things": {"type": "array"},\n                "count": {"type": "integer"},\n                "average": {"type": "number"},\n                "magic": {"type": "string"},\n                "flag": {"type": "boolean"},\n                "nothing": {"type": "null"},\n                "unknown_type": {"type": "world"},\n                "undefined_type": {},\n            }\n        }\n    '
    res = parse(code)
    assert isinstance(res, nodes.Module)
    class_node: nodes.ClassDef = res.body[0]
    assert isinstance(class_node, nodes.ClassDef)
    assert 'schema' in class_node.locals
    expected = {'thing': nodes.Dict, 'things': nodes.List, 'unknown_type': nodes.ClassDef, 'undefined_type': nodes.ClassDef}
    for (property_name, value_class) in expected.items():
        assert property_name in class_node.locals
        attribute_value_node = next(class_node.locals[property_name][0].infer())
        assert isinstance(attribute_value_node, value_class)
    expected = {'count': 'int', 'average': 'float', 'magic': 'str', 'flag': 'bool'}
    for (property_name, value_class_name) in expected.items():
        assert property_name in class_node.locals
        attribute_value_node = next(class_node.locals[property_name][0].infer())
        assert isinstance(attribute_value_node, nodes.ClassDef)
        assert attribute_value_node.name == value_class_name
    assert 'nothing' in class_node.locals
    attribute_value_node = next(class_node.locals['nothing'][0].infer())
    assert isinstance(attribute_value_node, nodes.Const)
    assert attribute_value_node.value is None

class TestTypeChecker(pylint.testutils.CheckerTestCase):
    CHECKER_CLASS = pylint.checkers.typecheck.TypeChecker
    checker: pylint.checkers.typecheck.TypeChecker

    def test_finds_no_member_on_api_model_when_property_not_in_schema(self):
        if False:
            for i in range(10):
                print('nop')
        (assign_node_present, assign_node_missing) = astroid.extract_node('\n            class TestAPI:\n                schema = {"properties": {"present": {"type": "string"}}}\n\n            def test():\n                model = TestAPI()\n                present = model.present  #@\n                missing = model.missing  #@\n            ')
        self.checker.visit_assign(assign_node_present)
        self.checker.visit_assign(assign_node_missing)
        with self.assertNoMessages():
            self.checker.visit_attribute(assign_node_present.value)
        with self.assertAddsMessages(pylint.testutils.Message(msg_id='no-member', args=('Instance of', 'TestAPI', 'missing', ''), node=assign_node_missing.value)):
            self.checker.visit_attribute(assign_node_missing.value)