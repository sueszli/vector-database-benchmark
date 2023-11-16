from __future__ import annotations
from tests.charts.helm_template_generator import render_chart

class TestPgbouncerPdb:
    """Tests PgBouncer PDB."""

    def test_should_pass_validation_with_just_pdb_enabled_v1(self):
        if False:
            i = 10
            return i + 15
        render_chart(values={'pgbouncer': {'enabled': True, 'podDisruptionBudget': {'enabled': True}}}, show_only=['templates/pgbouncer/pgbouncer-poddisruptionbudget.yaml'])

    def test_should_pass_validation_with_just_pdb_enabled_v1beta1(self):
        if False:
            print('Hello World!')
        render_chart(values={'pgbouncer': {'enabled': True, 'podDisruptionBudget': {'enabled': True}}}, show_only=['templates/pgbouncer/pgbouncer-poddisruptionbudget.yaml'], kubernetes_version='1.16.0')

    def test_should_pass_validation_with_pdb_enabled_and_min_available_param(self):
        if False:
            i = 10
            return i + 15
        render_chart(values={'pgbouncer': {'enabled': True, 'podDisruptionBudget': {'enabled': True, 'config': {'maxUnavailable': None, 'minAvailable': 1}}}}, show_only=['templates/pgbouncer/pgbouncer-poddisruptionbudget.yaml'])