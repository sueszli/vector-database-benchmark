from __future__ import annotations
import jmespath
from tests.charts.helm_template_generator import render_chart

class TestLogsPersistentVolumeClaim:
    """Tests logs PVC."""

    def test_should_not_generate_a_document_if_persistence_is_disabled(self):
        if False:
            return 10
        docs = render_chart(values={'logs': {'persistence': {'enabled': False}}}, show_only=['templates/logs-persistent-volume-claim.yaml'])
        assert 0 == len(docs)

    def test_should_not_generate_a_document_when_using_an_existing_claim(self):
        if False:
            print('Hello World!')
        docs = render_chart(values={'logs': {'persistence': {'enabled': True, 'existingClaim': 'test-claim'}}}, show_only=['templates/logs-persistent-volume-claim.yaml'])
        assert 0 == len(docs)

    def test_should_generate_a_document_if_persistence_is_enabled_and_not_using_an_existing_claim(self):
        if False:
            return 10
        docs = render_chart(values={'logs': {'persistence': {'enabled': True, 'existingClaim': None}}}, show_only=['templates/logs-persistent-volume-claim.yaml'])
        assert 1 == len(docs)

    def test_should_set_pvc_details_correctly(self):
        if False:
            i = 10
            return i + 15
        docs = render_chart(values={'logs': {'persistence': {'enabled': True, 'size': '1G', 'existingClaim': None, 'storageClassName': 'MyStorageClass'}}}, show_only=['templates/logs-persistent-volume-claim.yaml'])
        assert {'accessModes': ['ReadWriteMany'], 'resources': {'requests': {'storage': '1G'}}, 'storageClassName': 'MyStorageClass'} == jmespath.search('spec', docs[0])

    def test_logs_persistent_volume_claim_template_storage_class_name(self):
        if False:
            while True:
                i = 10
        docs = render_chart(values={'logs': {'persistence': {'existingClaim': None, 'enabled': True, 'storageClassName': '{{ .Release.Name }}-storage-class'}}}, show_only=['templates/logs-persistent-volume-claim.yaml'])
        assert 'release-name-storage-class' == jmespath.search('spec.storageClassName', docs[0])