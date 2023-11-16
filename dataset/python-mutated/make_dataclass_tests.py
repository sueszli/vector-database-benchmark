import ast
import os.path
import sys
unavailable_functions = frozenset({'dataclass_textanno', 'dataclass_module_1', 'make_dataclass'})
skip_tests = frozenset({('TestCase', 'test_field_default_default_factory_error'), ('TestCase', 'test_two_fields_one_default'), ('TestCase', 'test_overwrite_hash'), ('TestCase', 'test_eq_order'), ('TestCase', 'test_no_unhashable_default'), ('TestCase', 'test_disallowed_mutable_defaults'), ('TestCase', 'test_classvar_default_factory'), ('TestCase', 'test_field_metadata_mapping'), ('TestFieldNoAnnotation', 'test_field_without_annotation'), ('TestFieldNoAnnotation', 'test_field_without_annotation_but_annotation_in_base'), ('TestFieldNoAnnotation', 'test_field_without_annotation_but_annotation_in_base_not_dataclass'), ('TestOrdering', 'test_overwriting_order'), ('TestHash', 'test_hash_rules'), ('TestHash', 'test_hash_no_args'), ('TestFrozen', 'test_inherit_nonfrozen_from_empty_frozen'), ('TestFrozen', 'test_inherit_nonfrozen_from_frozen'), ('TestFrozen', 'test_inherit_frozen_from_nonfrozen'), ('TestFrozen', 'test_overwriting_frozen'), ('TestSlots', 'test_add_slots_when_slots_exists'), ('TestSlots', 'test_cant_inherit_from_iterator_slots'), ('TestSlots', 'test_weakref_slot_without_slot'), ('TestKeywordArgs', 'test_no_classvar_kwarg'), ('TestKeywordArgs', 'test_KW_ONLY_twice'), ('TestKeywordArgs', 'test_defaults'), ('TestCase', 'test_default_factory'), ('TestCase', 'test_default_factory_with_no_init'), ('TestCase', 'test_field_default'), ('TestCase', 'test_function_annotations'), ('TestDescriptors', 'test_lookup_on_instance'), ('TestCase', 'test_default_factory_not_called_if_value_given'), ('TestCase', 'test_class_attrs'), ('TestCase', 'test_hash_field_rules'), ('TestStringAnnotations',), ('TestOrdering', 'test_functools_total_ordering'), ('TestCase', 'test_missing_default_factory'), ('TestCase', 'test_missing_default'), ('TestCase', 'test_missing_repr'), ('TestSlots',), ('TestMatchArgs',), ('TestKeywordArgs', 'test_field_marked_as_kwonly'), ('TestKeywordArgs', 'test_match_args'), ('TestKeywordArgs', 'test_KW_ONLY'), ('TestKeywordArgs', 'test_KW_ONLY_as_string'), ('TestKeywordArgs', 'test_post_init'), ('TestCase', 'test_class_var_frozen'), ('TestCase', 'test_dont_include_other_annotations'), ('TestDocString',), ('TestCase', 'test_field_repr'), ('TestCase', 'test_dynamic_class_creation'), ('TestCase', 'test_dynamic_class_creation_using_field'), ('TestCase', 'test_is_dataclass_genericalias'), ('TestCase', 'test_generic_extending'), ('TestCase', 'test_generic_dataclasses'), ('TestCase', 'test_generic_dynamic'), ('TestInit', 'test_inherit_from_protocol'), ('TestAbstract', 'test_abc_implementation'), ('TestAbstract', 'test_maintain_abc'), ('TestCase', 'test_post_init_not_auto_added'), ('TestCase', 'test_post_init_staticmethod'), ('TestDescriptors', 'test_non_descriptor'), ('TestDescriptors', 'test_set_name'), ('TestDescriptors', 'test_setting_field_calls_set'), ('TestDescriptors', 'test_setting_uninitialized_descriptor_field'), ('TestCase', 'test_init_false_no_default'), ('TestCase', 'test_init_var_inheritance'), ('TestCase', 'test_base_has_init'), ('TestInit', 'test_base_has_init'), ('TestCase', 'test_post_init_super'), ('TestCase', 'test_init_in_order'), ('TestDescriptors', 'test_getting_field_calls_get'), ('TestDescriptors', 'test_init_calls_set'), ('TestHash', 'test_eq_only'), ('TestCase', 'test_items_in_dicts'), ('TestRepr', 'test_repr'), ('TestCase', 'test_not_in_repr'), ('TestRepr', 'test_no_repr'), ('TestInit', 'test_no_init'), ('TestOrdering', 'test_no_order'), ('TestCase', 'test_post_init_classmethod'), ('TestCase', 'test_field_order'), ('TestCase', 'test_overwrite_fields_in_derived_class'), ('TestCase', 'test_class_var'), ('TestFrozen',), ('TestCase', 'test_post_init'), ('TestReplace', 'test_frozen'), ('TestCase', 'test_dataclasses_qualnames'), ('TestCase', 'test_compare_subclasses'), ('TestCase', 'test_simple_compare'), ('TestCase', 'test_field_named_self'), ('TestCase', 'test_init_var_default_factory'), ('TestCase', 'test_init_var_no_default'), ('TestCase', 'test_init_var_with_default'), ('TestReplace', 'test_initvar_with_default_value'), ('TestCase', 'test_class_marker'), ('TestCase', 'test_field_metadata_custom_mapping'), ('TestCase', 'test_class_var_default_factory'), ('TestCase', 'test_class_var_with_default'), ('TestDescriptors',)})
version_specific_skips = {('TestCase', 'test_init_var_preserve_type'): (3, 10)}

class DataclassInDecorators(ast.NodeVisitor):
    found = False

    def visit_Name(self, node):
        if False:
            print('Hello World!')
        if node.id == 'dataclass':
            self.found = True
        return self.generic_visit(node)

    def generic_visit(self, node):
        if False:
            while True:
                i = 10
        if self.found:
            return
        return super().generic_visit(node)

def dataclass_in_decorators(decorator_list):
    if False:
        i = 10
        return i + 15
    finder = DataclassInDecorators()
    for dec in decorator_list:
        finder.visit(dec)
        if finder.found:
            return True
    return False

class SubstituteNameString(ast.NodeTransformer):

    def __init__(self, substitutions):
        if False:
            print('Hello World!')
        super().__init__()
        self.substitutions = substitutions

    def visit_Constant(self, node):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node.value, str):
            if node.value.find('<locals>') != -1:
                import re
                new_value = new_value2 = re.sub('[\\w.]*<locals>', '', node.value)
                for (key, value) in self.substitutions.items():
                    new_value2 = re.sub(f'(?<![\\w])[.]{key}(?![\\w])', value, new_value2)
                if new_value != new_value2:
                    node.value = new_value2
        return node

class SubstituteName(SubstituteNameString):

    def visit_Name(self, node):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node.ctx, ast.Store):
            return node
        replacement = self.substitutions.get(node.id, None)
        if replacement is not None:
            return ast.Name(id=replacement, ctx=node.ctx)
        else:
            return node

class IdentifyCdefClasses(ast.NodeVisitor):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.top_level_class = True
        self.classes = {}
        self.cdef_classes = set()

    def visit_ClassDef(self, node):
        if False:
            print('Hello World!')
        (top_level_class, self.top_level_class) = (self.top_level_class, False)
        try:
            if not top_level_class:
                self.classes[node.name] = node
                if dataclass_in_decorators(node.decorator_list):
                    self.handle_cdef_class(node)
                self.generic_visit(node)
            else:
                self.generic_visit(node)
        finally:
            self.top_level_class = top_level_class

    def visit_FunctionDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        (classes, self.classes) = (self.classes, {})
        self.generic_visit(node)
        self.classes = classes

    def handle_cdef_class(self, cls_node):
        if False:
            while True:
                i = 10
        if cls_node not in self.cdef_classes:
            self.cdef_classes.add(cls_node)
            if cls_node.bases and isinstance(cls_node.bases[0], ast.Name):
                base0_node = self.classes.get(cls_node.bases[0].id)
                if base0_node:
                    self.handle_cdef_class(base0_node)

class ExtractDataclassesToTopLevel(ast.NodeTransformer):

    def __init__(self, cdef_classes_set):
        if False:
            print('Hello World!')
        super().__init__()
        self.nested_name = []
        self.current_function_global_classes = []
        self.global_classes = []
        self.cdef_classes_set = cdef_classes_set
        self.used_names = set()
        self.collected_substitutions = {}
        self.uses_unavailable_name = False
        self.top_level_class = True

    def visit_ClassDef(self, node):
        if False:
            print('Hello World!')
        if not self.top_level_class:
            self.generic_visit(node)
            if not node.body:
                node.body.append(ast.Pass)
            if node in self.cdef_classes_set:
                node.decorator_list.append(ast.Name(id='cclass', ctx=ast.Load()))
            old_name = node.name
            new_name = '_'.join([node.name] + self.nested_name)
            while new_name in self.used_names:
                new_name = new_name + '_'
            node.name = new_name
            self.current_function_global_classes.append(node)
            self.used_names.add(new_name)
            self.collected_substitutions[old_name] = node.name
            return ast.Assign(targets=[ast.Name(id=old_name, ctx=ast.Store())], value=ast.Name(id=new_name, ctx=ast.Load()), lineno=-1)
        else:
            (top_level_class, self.top_level_class) = (self.top_level_class, False)
            self.nested_name.append(node.name)
            if tuple(self.nested_name) in skip_tests:
                self.top_level_class = top_level_class
                self.nested_name.pop()
                return None
            self.generic_visit(node)
            self.nested_name.pop()
            if not node.body:
                node.body.append(ast.Pass())
            self.top_level_class = top_level_class
            return node

    def visit_FunctionDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.nested_name.append(node.name)
        if tuple(self.nested_name) in skip_tests:
            self.nested_name.pop()
            return None
        if tuple(self.nested_name) in version_specific_skips:
            version = version_specific_skips[tuple(self.nested_name)]
            decorator = ast.parse(f'skip_on_versions_below({version})', mode='eval').body
            node.decorator_list.append(decorator)
        (collected_subs, self.collected_substitutions) = (self.collected_substitutions, {})
        (uses_unavailable_name, self.uses_unavailable_name) = (self.uses_unavailable_name, False)
        (current_func_globs, self.current_function_global_classes) = (self.current_function_global_classes, [])
        self.generic_visit(node)
        if self.collected_substitutions:
            node = SubstituteNameString(self.collected_substitutions).visit(node)
            replacer = SubstituteName(self.collected_substitutions)
            for global_class in self.current_function_global_classes:
                global_class = replacer.visit(global_class)
        self.global_classes.append(self.current_function_global_classes)
        self.nested_name.pop()
        self.collected_substitutions = collected_subs
        if self.uses_unavailable_name:
            node = None
        self.uses_unavailable_name = uses_unavailable_name
        self.current_function_global_classes = current_func_globs
        return node

    def visit_Name(self, node):
        if False:
            while True:
                i = 10
        if node.id in unavailable_functions:
            self.uses_unavailable_name = True
        return self.generic_visit(node)

    def visit_Import(self, node):
        if False:
            print('Hello World!')
        return None

    def visit_ImportFrom(self, node):
        if False:
            return 10
        return None

    def visit_Call(self, node):
        if False:
            return 10
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'assertRaisesRegex':
            node.func.attr = 'assertRaises'
            node.args.pop()
        return self.generic_visit(node)

    def visit_Module(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.generic_visit(node)
        node.body[0:0] = self.global_classes
        return node

    def visit_AnnAssign(self, node):
        if False:
            while True:
                i = 10
        if isinstance(node.annotation, ast.Constant) and isinstance(node.annotation.value, str):
            node.annotation = ast.Name(id='object', ctx=ast.Load)
        return node

def main():
    if False:
        while True:
            i = 10
    script_path = os.path.split(sys.argv[0])[0]
    filename = 'test_dataclasses.py'
    py_module_path = os.path.join(script_path, 'dataclass_test_data', filename)
    with open(py_module_path, 'r') as f:
        tree = ast.parse(f.read(), filename)
    cdef_class_finder = IdentifyCdefClasses()
    cdef_class_finder.visit(tree)
    transformer = ExtractDataclassesToTopLevel(cdef_class_finder.cdef_classes)
    tree = transformer.visit(tree)
    output_path = os.path.join(script_path, '..', 'tests', 'run', filename + 'x')
    with open(output_path, 'w') as f:
        print('# AUTO-GENERATED BY Tools/make_dataclass_tests.py', file=f)
        print('# DO NOT EDIT', file=f)
        print(file=f)
        print('# cython: language_level=3', file=f)
        print('include "test_dataclasses.pxi"', file=f)
        print(file=f)
        print(ast.unparse(tree), file=f)
if __name__ == '__main__':
    main()