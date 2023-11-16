from __future__ import annotations
import jmespath
from tests.charts.helm_template_generator import render_chart

class TestDagsPersistentVolumeClaim:
    """Tests DAGs PVC."""

    def test_should_not_generate_a_document_if_persistence_is_disabled(self):
        if False:
            return 10
        docs = render_chart(values={'dags': {'persistence': {'enabled': False}}}, show_only=['templates/dags-persistent-volume-claim.yaml'])
        assert 0 == len(docs)

    def test_should_not_generate_a_document_when_using_an_existing_claim(self):
        if False:
            i = 10
            return i + 15
        docs = render_chart(values={'dags': {'persistence': {'enabled': True, 'existingClaim': 'test-claim'}}}, show_only=['templates/dags-persistent-volume-claim.yaml'])
        assert 0 == len(docs)

    def test_should_generate_a_document_if_persistence_is_enabled_and_not_using_an_existing_claim(self):
        if False:
            i = 10
            return i + 15
        docs = render_chart(values={'dags': {'persistence': {'enabled': True, 'existingClaim': None}}}, show_only=['templates/dags-persistent-volume-claim.yaml'])
        assert 1 == len(docs)

    def test_should_set_pvc_details_correctly(self):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'dags': {'persistence': {'enabled': True, 'size': '1G', 'existingClaim': None, 'storageClassName': 'MyStorageClass', 'accessMode': 'ReadWriteMany'}}}, show_only=['templates/dags-persistent-volume-claim.yaml'])
        assert {'accessModes': ['ReadWriteMany'], 'resources': {'requests': {'storage': '1G'}}, 'storageClassName': 'MyStorageClass'} == jmespath.search('spec', docs[0])

    def test_single_annotation(self):
        if False:
            return 10
        docs = render_chart(values={'dags': {'persistence': {'enabled': True, 'size': '1G', 'existingClaim': None, 'storageClassName': 'MyStorageClass', 'accessMode': 'ReadWriteMany', 'annotations': {'key': 'value'}}}}, show_only=['templates/dags-persistent-volume-claim.yaml'])
        annotations = jmespath.search('metadata.annotations', docs[0])
        assert 'value' == annotations.get('key')

    def test_multiple_annotations(self):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'dags': {'persistence': {'enabled': True, 'size': '1G', 'existingClaim': None, 'storageClassName': 'MyStorageClass', 'accessMode': 'ReadWriteMany', 'annotations': {'key': 'value', 'key-two': 'value-two'}}}}, show_only=['templates/dags-persistent-volume-claim.yaml'])
        annotations = jmespath.search('metadata.annotations', docs[0])
        assert 'value' == annotations.get('key')
        assert 'value-two' == annotations.get('key-two')

    def test_dags_persistent_volume_claim_template_storage_class_name(self):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'dags': {'persistence': {'existingClaim': None, 'enabled': True, 'storageClassName': '{{ .Release.Name }}-storage-class'}}}, show_only=['templates/dags-persistent-volume-claim.yaml'])
        assert 'release-name-storage-class' == jmespath.search('spec.storageClassName', docs[0])