from __future__ import annotations
import itertools
import jmespath
import pytest
from tests.charts.helm_template_generator import render_chart

class TestIngressFlower:
    """Tests ingress flower."""

    def test_should_pass_validation_with_just_ingress_enabled_v1(self):
        if False:
            for i in range(10):
                print('nop')
        render_chart(values={'flower': {'enabled': True}, 'ingress': {'flower': {'enabled': True}}}, show_only=['templates/flower/flower-ingress.yaml'])

    def test_should_pass_validation_with_just_ingress_enabled_v1beta1(self):
        if False:
            for i in range(10):
                print('nop')
        render_chart(values={'flower': {'enabled': True}, 'ingress': {'flower': {'enabled': True}}}, show_only=['templates/flower/flower-ingress.yaml'], kubernetes_version='1.16.0')

    def test_should_allow_more_than_one_annotation(self):
        if False:
            print('Hello World!')
        docs = render_chart(values={'ingress': {'flower': {'enabled': True, 'annotations': {'aa': 'bb', 'cc': 'dd'}}}, 'flower': {'enabled': True}}, show_only=['templates/flower/flower-ingress.yaml'])
        assert jmespath.search('metadata.annotations', docs[0]) == {'aa': 'bb', 'cc': 'dd'}

    def test_should_set_ingress_class_name(self):
        if False:
            i = 10
            return i + 15
        docs = render_chart(values={'ingress': {'enabled': True, 'flower': {'ingressClassName': 'foo'}}, 'flower': {'enabled': True}}, show_only=['templates/flower/flower-ingress.yaml'])
        assert 'foo' == jmespath.search('spec.ingressClassName', docs[0])

    def test_should_ingress_hosts_objs_have_priority_over_host(self):
        if False:
            while True:
                i = 10
        docs = render_chart(values={'flower': {'enabled': True}, 'ingress': {'flower': {'enabled': True, 'tls': {'enabled': True, 'secretName': 'oldsecret'}, 'hosts': [{'name': '*.a-host', 'tls': {'enabled': True, 'secretName': 'newsecret1'}}, {'name': 'b-host', 'tls': {'enabled': True, 'secretName': 'newsecret2'}}, {'name': 'c-host', 'tls': {'enabled': True, 'secretName': 'newsecret1'}}, {'name': 'd-host', 'tls': {'enabled': False, 'secretName': ''}}, {'name': 'e-host'}], 'host': 'old-host'}}}, show_only=['templates/flower/flower-ingress.yaml'])
        assert ['*.a-host', 'b-host', 'c-host', 'd-host', 'e-host'] == jmespath.search('spec.rules[*].host', docs[0])
        assert [{'hosts': ['*.a-host'], 'secretName': 'newsecret1'}, {'hosts': ['b-host'], 'secretName': 'newsecret2'}, {'hosts': ['c-host'], 'secretName': 'newsecret1'}] == jmespath.search('spec.tls[*]', docs[0])

    def test_should_ingress_hosts_strs_have_priority_over_host(self):
        if False:
            return 10
        docs = render_chart(values={'flower': {'enabled': True}, 'ingress': {'flower': {'enabled': True, 'tls': {'enabled': True, 'secretName': 'secret'}, 'hosts': ['*.a-host', 'b-host', 'c-host', 'd-host'], 'host': 'old-host'}}}, show_only=['templates/flower/flower-ingress.yaml'])
        assert ['*.a-host', 'b-host', 'c-host', 'd-host'] == jmespath.search('spec.rules[*].host', docs[0])
        assert [{'hosts': ['*.a-host', 'b-host', 'c-host', 'd-host'], 'secretName': 'secret'}] == jmespath.search('spec.tls[*]', docs[0])

    def test_should_ingress_deprecated_host_and_top_level_tls_still_work(self):
        if False:
            print('Hello World!')
        docs = render_chart(values={'flower': {'enabled': True}, 'ingress': {'flower': {'enabled': True, 'tls': {'enabled': True, 'secretName': 'supersecret'}, 'host': 'old-host'}}}, show_only=['templates/flower/flower-ingress.yaml'])
        assert ['old-host'] == jmespath.search('spec.rules[*].host', docs[0]) == list(itertools.chain.from_iterable(jmespath.search('spec.tls[*].hosts', docs[0])))

    def test_should_ingress_host_entry_not_exist(self):
        if False:
            return 10
        docs = render_chart(values={'flower': {'enabled': True}, 'ingress': {'flower': {'enabled': True}}}, show_only=['templates/flower/flower-ingress.yaml'])
        assert not jmespath.search('spec.rules[*].host', docs[0])

    @pytest.mark.parametrize('global_value, flower_value, expected', [(None, None, False), (None, False, False), (None, True, True), (False, None, False), (True, None, True), (False, True, True), (True, False, True)])
    def test_ingress_created(self, global_value, flower_value, expected):
        if False:
            while True:
                i = 10
        values = {'flower': {'enabled': True}, 'ingress': {}}
        if global_value is not None:
            values['ingress']['enabled'] = global_value
        if flower_value is not None:
            values['ingress']['flower'] = {'enabled': flower_value}
        if values['ingress'] == {}:
            del values['ingress']
        docs = render_chart(values=values, show_only=['templates/flower/flower-ingress.yaml'])
        assert expected == (1 == len(docs))

    def test_ingress_not_created_flower_disabled(self):
        if False:
            print('Hello World!')
        docs = render_chart(values={'ingress': {'flower': {'enabled': True}}}, show_only=['templates/flower/flower-ingress.yaml'])
        assert 0 == len(docs)

    def test_should_add_component_specific_labels(self):
        if False:
            while True:
                i = 10
        docs = render_chart(values={'ingress': {'enabled': True}, 'flower': {'enabled': True, 'labels': {'test_label': 'test_label_value'}}}, show_only=['templates/flower/flower-ingress.yaml'])
        assert 'test_label' in jmespath.search('metadata.labels', docs[0])
        assert jmespath.search('metadata.labels', docs[0])['test_label'] == 'test_label_value'

    def test_can_ingress_hosts_be_templated(self):
        if False:
            i = 10
            return i + 15
        docs = render_chart(values={'testValues': {'scalar': 'aa', 'list': ['bb', 'cc'], 'dict': {'key': 'dd'}}, 'flower': {'enabled': True}, 'ingress': {'flower': {'enabled': True, 'hosts': [{'name': '*.{{ .Release.Namespace }}.example.com'}, {'name': '{{ .Values.testValues.scalar }}.example.com'}, {'name': '{{ index .Values.testValues.list 1 }}.example.com'}, {'name': '{{ .Values.testValues.dict.key }}.example.com'}]}}}, show_only=['templates/flower/flower-ingress.yaml'], namespace='airflow')
        assert ['*.airflow.example.com', 'aa.example.com', 'cc.example.com', 'dd.example.com'] == jmespath.search('spec.rules[*].host', docs[0])