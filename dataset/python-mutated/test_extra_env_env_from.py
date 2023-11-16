from __future__ import annotations
import textwrap
from typing import Any
import jmespath
import pytest
import yaml
from tests.charts.helm_template_generator import prepare_k8s_lookup_dict, render_chart
RELEASE_NAME = 'test-extra-env-env-from'
PARAMS = [(('Job', f'{RELEASE_NAME}-create-user'), ('spec.template.spec.containers[0]',)), (('Job', f'{RELEASE_NAME}-run-airflow-migrations'), ('spec.template.spec.containers[0]',)), (('Deployment', f'{RELEASE_NAME}-scheduler'), ('spec.template.spec.initContainers[0]', 'spec.template.spec.containers[0]')), (('StatefulSet', f'{RELEASE_NAME}-worker'), ('spec.template.spec.initContainers[0]', 'spec.template.spec.containers[0]')), (('Deployment', f'{RELEASE_NAME}-webserver'), ('spec.template.spec.initContainers[0]', 'spec.template.spec.containers[0]')), (('StatefulSet', f'{RELEASE_NAME}-triggerer'), ('spec.template.spec.initContainers[0]', 'spec.template.spec.containers[0]')), (('Deployment', f'{RELEASE_NAME}-flower'), ('spec.template.spec.containers[0]',))]

class TestExtraEnvEnvFrom:
    """Tests extra env from."""
    k8s_objects: list[dict[str, Any]]
    k8s_objects_by_key: dict[tuple[str, str], dict[str, Any]]

    @classmethod
    def setup_class(cls) -> None:
        if False:
            while True:
                i = 10
        values_str = textwrap.dedent('\n            airflowVersion: "2.6.0"\n            flower:\n              enabled: true\n            extraEnvFrom: |\n              - secretRef:\n                  name: \'{{ .Release.Name }}-airflow-connections\'\n              - configMapRef:\n                  name: \'{{ .Release.Name }}-airflow-variables\'\n            extraEnv: |\n              - name: PLATFORM\n                value: FR\n              - name: TEST\n                valueFrom:\n                  secretKeyRef:\n                    name: \'{{ .Release.Name }}-some-secret\'\n                    key: connection\n            ')
        values = yaml.safe_load(values_str)
        cls.k8s_objects = render_chart(RELEASE_NAME, values=values)
        cls.k8s_objects_by_key = prepare_k8s_lookup_dict(cls.k8s_objects)

    @pytest.mark.parametrize('k8s_obj_key, env_paths', PARAMS)
    def test_extra_env(self, k8s_obj_key, env_paths):
        if False:
            return 10
        expected_env_as_str = textwrap.dedent(f'\n            - name: PLATFORM\n              value: FR\n            - name: TEST\n              valueFrom:\n                secretKeyRef:\n                  key: connection\n                  name: {RELEASE_NAME}-some-secret\n            ').lstrip()
        k8s_object = self.k8s_objects_by_key[k8s_obj_key]
        for path in env_paths:
            env = jmespath.search(f'{path}.env', k8s_object)
            assert expected_env_as_str in yaml.dump(env)

    @pytest.mark.parametrize('k8s_obj_key, env_from_paths', PARAMS)
    def test_extra_env_from(self, k8s_obj_key, env_from_paths):
        if False:
            print('Hello World!')
        expected_env_from_as_str = textwrap.dedent(f'\n            - secretRef:\n                name: {RELEASE_NAME}-airflow-connections\n            - configMapRef:\n                name: {RELEASE_NAME}-airflow-variables\n            ').lstrip()
        k8s_object = self.k8s_objects_by_key[k8s_obj_key]
        for path in env_from_paths:
            env_from = jmespath.search(f'{path}.envFrom', k8s_object)
            assert expected_env_from_as_str in yaml.dump(env_from)