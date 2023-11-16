from __future__ import annotations
import jmespath
from tests.charts.helm_template_generator import render_chart

class TestSchedulerPdb:
    """Tests Scheduler PDB."""

    def test_should_pass_validation_with_just_pdb_enabled_v1(self):
        if False:
            while True:
                i = 10
        render_chart(values={'scheduler': {'podDisruptionBudget': {'enabled': True}}}, show_only=['templates/scheduler/scheduler-poddisruptionbudget.yaml'])

    def test_should_pass_validation_with_just_pdb_enabled_v1beta1(self):
        if False:
            while True:
                i = 10
        render_chart(values={'scheduler': {'podDisruptionBudget': {'enabled': True}}}, show_only=['templates/scheduler/scheduler-poddisruptionbudget.yaml'], kubernetes_version='1.16.0')

    def test_should_add_component_specific_labels(self):
        if False:
            i = 10
            return i + 15
        docs = render_chart(values={'scheduler': {'podDisruptionBudget': {'enabled': True}, 'labels': {'test_label': 'test_label_value'}}}, show_only=['templates/scheduler/scheduler-poddisruptionbudget.yaml'])
        assert 'test_label' in jmespath.search('metadata.labels', docs[0])
        assert jmespath.search('metadata.labels', docs[0])['test_label'] == 'test_label_value'

    def test_should_pass_validation_with_pdb_enabled_and_min_available_param(self):
        if False:
            i = 10
            return i + 15
        render_chart(values={'scheduler': {'podDisruptionBudget': {'enabled': True, 'config': {'maxUnavailable': None, 'minAvailable': 1}}}}, show_only=['templates/scheduler/scheduler-poddisruptionbudget.yaml'])