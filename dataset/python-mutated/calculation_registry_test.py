"""Tests for calculation registry."""
from __future__ import annotations
from core.domain import calculation_registry
from core.tests import test_utils
from extensions.answer_summarizers import models

class CalculationRegistryTests(test_utils.GenericTestBase):
    """Provides testing of the calculation registry."""

    def test_get_calculation_by_id(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(isinstance(calculation_registry.Registry.get_calculation_by_id('AnswerFrequencies'), models.AnswerFrequencies))
        with self.assertRaisesRegex(TypeError, "'a' is not a valid calculation id."):
            calculation_registry.Registry.get_calculation_by_id('a')

    def test_get_calculation_by_id_when_calculations_dict_have_calculation_id(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertTrue(isinstance(calculation_registry.Registry.get_calculation_by_id('Top5AnswerFrequencies'), models.Top5AnswerFrequencies))
        self.assertTrue(isinstance(calculation_registry.Registry.get_calculation_by_id('Top5AnswerFrequencies'), models.Top5AnswerFrequencies))