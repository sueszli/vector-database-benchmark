"""Unit tests for core.domain.customization_args_utils."""
from __future__ import annotations
import os
import re
from core import feconf
from core import utils
from core.domain import customization_args_util
from core.domain import interaction_registry
from core.tests import test_utils
from typing import Dict, List, Union

class CustomizationArgsUtilUnitTests(test_utils.GenericTestBase):
    """Test customization args generation and validation."""

    def test_validate_customization_args_and_values(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test validate customization args and values method.'
        ca_item_selection_specs = interaction_registry.Registry.get_interaction_by_id('ItemSelectionInput').customization_arg_specs
        complete_customization_args: Dict[str, Dict[str, Union[int, List[str]]]] = {'minAllowableSelectionCount': {'value': 1}, 'maxAllowableSelectionCount': {'value': 1}, 'choices': {'value': ['']}}
        complete_customization_args_with_invalid_arg_name = {'minAllowableSelectionCount': {'value': 1}, 'maxAllowableSelectionCount': {'value': 1}, 'choices': {'value': ['']}, 23: {'value': ''}}
        complete_customization_args_with_extra_arg: Dict[str, Dict[str, Union[int, str, List[str]]]] = {'minAllowableSelectionCount': {'value': 1}, 'maxAllowableSelectionCount': {'value': 1}, 'choices': {'value': ['']}, 'extraArg': {'value': ''}}
        complete_customization_args_with_invalid_arg_type: Dict[str, Dict[str, Union[str, int, List[str]]]] = {'minAllowableSelectionCount': {'value': 'invalid'}, 'maxAllowableSelectionCount': {'value': 1}, 'choices': {'value': ['']}}
        expected_customization_args_after_validation = {'minAllowableSelectionCount': {'value': 1}, 'maxAllowableSelectionCount': {'value': 1}, 'choices': {'value': ['']}}
        expected_customization_args_after_validation_with_invalid_arg_type = complete_customization_args_with_invalid_arg_type
        customization_args_util.validate_customization_args_and_values('interaction', 'ItemSelectionInput', complete_customization_args, ca_item_selection_specs)
        self.assertEqual(expected_customization_args_after_validation, complete_customization_args)
        with self.assertRaisesRegex(utils.ValidationError, 'Invalid customization arg name: 23'):
            customization_args_util.validate_customization_args_and_values('interaction', 'ItemSelectionInput', complete_customization_args_with_invalid_arg_name, ca_item_selection_specs)
        with self.assertRaisesRegex(utils.ValidationError, 'Interaction ItemSelectionInput does not support customization arg extraArg.'):
            customization_args_util.validate_customization_args_and_values('interaction', 'ItemSelectionInput', complete_customization_args_with_extra_arg, ca_item_selection_specs)
        customization_args_util.validate_customization_args_and_values('interaction', 'ItemSelectionInput', complete_customization_args_with_invalid_arg_type, ca_item_selection_specs)
        self.assertEqual(expected_customization_args_after_validation_with_invalid_arg_type, complete_customization_args_with_invalid_arg_type)
        ca_fraction_input_specs = interaction_registry.Registry.get_interaction_by_id('FractionInput').customization_arg_specs
        incomplete_customization_args = {'requireSimplestForm': {'value': False}, 'allowNonzeroIntegerPart': {'value': False}}
        incomplete_customization_args_with_invalid_arg_name = {'requireSimplestForm': {'value': False}, False: {'value': False}}
        complete_customization_args_with_invalid_arg_type = {'requireSimplestForm': {'value': False}, 'allowImproperFraction': {'value': True}, 'allowNonzeroIntegerPart': {'value': False}, 'customPlaceholder': {'value': 12}}
        complete_customization_args_with_extra_arg = {'requireSimplestForm': {'value': False}, 'allowImproperFraction': {'value': True}, 'allowNonzeroIntegerPart': {'value': False}, 'customPlaceholder': {'value': ''}, 'extraArg': {'value': ''}}
        with self.assertRaisesRegex(utils.ValidationError, 'Customization argument is missing key: allowImproperFraction'):
            customization_args_util.validate_customization_args_and_values('interaction', 'FractionInput', incomplete_customization_args, ca_fraction_input_specs)
        with self.assertRaisesRegex(utils.ValidationError, 'Invalid customization arg name: False'):
            customization_args_util.validate_customization_args_and_values('interaction', 'FractionInput', incomplete_customization_args_with_invalid_arg_name, ca_fraction_input_specs)
        with self.assertRaisesRegex(utils.ValidationError, 'Interaction FractionInput does not support customization arg extraArg.'):
            customization_args_util.validate_customization_args_and_values('interaction', 'FractionInput', complete_customization_args_with_extra_arg, ca_fraction_input_specs)
        customization_args_util.validate_customization_args_and_values('interaction', 'FractionInput', complete_customization_args_with_invalid_arg_type, ca_fraction_input_specs)
        self.assertEqual(complete_customization_args_with_invalid_arg_type, {'requireSimplestForm': {'value': False}, 'allowImproperFraction': {'value': True}, 'allowNonzeroIntegerPart': {'value': False}, 'customPlaceholder': {'value': 12}})
        customization_args_with_invalid_type = 23
        with self.assertRaisesRegex(utils.ValidationError, 'Expected customization args to be a dict, received %s' % customization_args_with_invalid_type):
            customization_args_util.validate_customization_args_and_values('interaction', 'FractionInput', customization_args_with_invalid_type, ca_fraction_input_specs)

    def test_validate_customization_args_and_values_with_invalid_schema(self) -> None:
        if False:
            while True:
                i = 10
        'Test validate customization args and values method with\n        invalid schema and errors raised on validation failure.\n        '
        ca_item_selection_specs = interaction_registry.Registry.get_interaction_by_id('ItemSelectionInput').customization_arg_specs
        invalid_customization_args: Dict[str, Dict[str, Union[str, int, List[str]]]] = {'minAllowableSelectionCount': {'value': '1b'}, 'maxAllowableSelectionCount': {'value': 1}, 'choices': {'value': ['']}}
        with self.assertRaisesRegex(utils.ValidationError, 'Could not convert str to int: 1b'):
            customization_args_util.validate_customization_args_and_values('interaction', 'ItemSelectionInput', invalid_customization_args, ca_item_selection_specs, fail_on_validation_errors=True)

    def test_frontend_customization_args_defs_coverage(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Test to ensure that customization-args-defs.ts has both frontend and\n        backend interfaces for each interaction's customization arguments.\n\n        Specifically: given an interaction with id 'X', there must exist an\n        interface in customization-args-defs.ts named XCustomizationArgs and\n        XCustomizationArgsBackendDict.\n        "
        filepath = os.path.join(feconf.INTERACTIONS_DIR, 'customization-args-defs.ts')
        with utils.open_file(filepath, 'r', newline='') as f:
            lines = f.readlines()
        all_interaction_ids = set(interaction_registry.Registry.get_all_interaction_ids())
        interaction_ids_with_ca_backend_interfaces = set()
        interaction_ids_with_ca_frontend_interfaces = set()
        for line in lines:
            ca_backend_interface_match = re.search('(interface )([a-zA-Z]+)(CustomizationArgsBackendDict)', line)
            if ca_backend_interface_match:
                interaction_ids_with_ca_backend_interfaces.add(ca_backend_interface_match.group(2))
            ca_frontend_interface_match = re.search('(interface )([a-zA-Z]+)(CustomizationArgs)( |{)', line)
            if ca_frontend_interface_match:
                interaction_ids_with_ca_frontend_interfaces.add(ca_frontend_interface_match.group(2))
        self.assertGreater(len(interaction_ids_with_ca_backend_interfaces), 0)
        self.assertEqual(all_interaction_ids, interaction_ids_with_ca_backend_interfaces)
        self.assertGreater(len(interaction_ids_with_ca_frontend_interfaces), 0)
        self.assertEqual(all_interaction_ids, interaction_ids_with_ca_frontend_interfaces)

    def test_frontend_customization_args_constructor_coverage(self) -> None:
        if False:
            while True:
                i = 10
        'Test to ensure that InteractionObjectFactory.ts covers constructing\n        customization arguments for each interaction. Uses regex to confirm\n        that the CustomizationArgs or CustomizationArgsBackendDict\n        interface is used in the file to typecast customization arguments.\n        '
        filepath = os.path.join('core', 'templates', 'domain', 'exploration', 'InteractionObjectFactory.ts')
        with utils.open_file(filepath, 'r', newline='') as f:
            lines = f.readlines()
        all_interaction_ids = set(interaction_registry.Registry.get_all_interaction_ids())
        interaction_ids_with_used_ca_frontend_interfaces = set()
        for line in lines:
            used_match = re.search('(as )([a-zA-Z]+)(CustomizationArgs)(BackendDict)?', line)
            if used_match:
                interaction_ids_with_used_ca_frontend_interfaces.add(used_match.group(2))
        self.assertEqual(all_interaction_ids, interaction_ids_with_used_ca_frontend_interfaces)

    def test_frontend_customization_args_dtslint_test_coverage(self) -> None:
        if False:
            print('Hello World!')
        'Test to ensure that customization-args-defs-test.ts covers testing\n        customization arguments types for each interaction. Uses regex to\n        confirm that there exists a test named\n        Test[interaction id]CustomizationArgsInterfacesMatch for each\n        interaction id.\n        '
        filepath = os.path.join('typings', 'tests', 'customization-args-defs-test.ts')
        with utils.open_file(filepath, 'r', newline='') as f:
            lines = f.readlines()
        all_interaction_ids = set(interaction_registry.Registry.get_all_interaction_ids())
        interaction_ids_with_ca_tests = set()
        for line in lines:
            test_exists_match = re.search('(Test)([a-zA-Z]+)(CustomizationArgsInterfacesMatch)', line)
            if test_exists_match:
                interaction_ids_with_ca_tests.add(test_exists_match.group(2))
        self.assertEqual(all_interaction_ids, interaction_ids_with_ca_tests)