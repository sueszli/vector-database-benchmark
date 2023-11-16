from __future__ import annotations
from copy import deepcopy
import pytest
from airflow.providers.google.cloud.utils.field_sanitizer import GcpBodyFieldSanitizer

class TestGcpBodyFieldSanitizer:

    def test_sanitize_should_sanitize_empty_body_and_fields(self):
        if False:
            while True:
                i = 10
        body = {}
        fields_to_sanitize = []
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        sanitizer.sanitize(body)
        assert {} == body

    def test_sanitize_should_not_fail_with_none_body(self):
        if False:
            return 10
        body = None
        fields_to_sanitize = []
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        sanitizer.sanitize(body)
        assert body is None

    def test_sanitize_should_fail_with_none_fields(self):
        if False:
            while True:
                i = 10
        body = {}
        fields_to_sanitize = None
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        with pytest.raises(TypeError):
            sanitizer.sanitize(body)

    def test_sanitize_should_not_fail_if_field_is_absent_in_body(self):
        if False:
            return 10
        body = {}
        fields_to_sanitize = ['kind']
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        sanitizer.sanitize(body)
        assert {} == body

    def test_sanitize_should_not_remove_fields_for_incorrect_specification(self):
        if False:
            i = 10
            return i + 15
        actual_body = [{'kind': 'compute#instanceTemplate', 'name': 'instance'}, {'kind': 'compute#instanceTemplate1', 'name': 'instance1'}, {'kind': 'compute#instanceTemplate2', 'name': 'instance2'}]
        body = deepcopy(actual_body)
        fields_to_sanitize = ['kind']
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        sanitizer.sanitize(body)
        assert actual_body == body

    def test_sanitize_should_remove_all_fields_from_root_level(self):
        if False:
            for i in range(10):
                print('nop')
        body = {'kind': 'compute#instanceTemplate', 'name': 'instance'}
        fields_to_sanitize = ['kind']
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        sanitizer.sanitize(body)
        assert {'name': 'instance'} == body

    def test_sanitize_should_remove_for_multiple_fields_from_root_level(self):
        if False:
            i = 10
            return i + 15
        body = {'kind': 'compute#instanceTemplate', 'name': 'instance'}
        fields_to_sanitize = ['kind', 'name']
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        sanitizer.sanitize(body)
        assert {} == body

    def test_sanitize_should_remove_all_fields_in_a_list_value(self):
        if False:
            for i in range(10):
                print('nop')
        body = {'fields': [{'kind': 'compute#instanceTemplate', 'name': 'instance'}, {'kind': 'compute#instanceTemplate1', 'name': 'instance1'}, {'kind': 'compute#instanceTemplate2', 'name': 'instance2'}]}
        fields_to_sanitize = ['fields.kind']
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        sanitizer.sanitize(body)
        assert {'fields': [{'name': 'instance'}, {'name': 'instance1'}, {'name': 'instance2'}]} == body

    def test_sanitize_should_remove_all_fields_in_any_nested_body(self):
        if False:
            while True:
                i = 10
        fields_to_sanitize = ['kind', 'properties.disks.kind', 'properties.metadata.kind']
        body = {'kind': 'compute#instanceTemplate', 'name': 'instance', 'properties': {'disks': [{'name': 'a', 'kind': 'compute#attachedDisk', 'type': 'PERSISTENT', 'mode': 'READ_WRITE'}, {'name': 'b', 'kind': 'compute#attachedDisk', 'type': 'PERSISTENT', 'mode': 'READ_WRITE'}], 'metadata': {'kind': 'compute#metadata', 'fingerprint': 'GDPUYxlwHe4='}}}
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        sanitizer.sanitize(body)
        assert {'name': 'instance', 'properties': {'disks': [{'name': 'a', 'type': 'PERSISTENT', 'mode': 'READ_WRITE'}, {'name': 'b', 'type': 'PERSISTENT', 'mode': 'READ_WRITE'}], 'metadata': {'fingerprint': 'GDPUYxlwHe4='}}} == body

    def test_sanitize_should_not_fail_if_specification_has_none_value(self):
        if False:
            i = 10
            return i + 15
        fields_to_sanitize = ['kind', 'properties.disks.kind', 'properties.metadata.kind']
        body = {'kind': 'compute#instanceTemplate', 'name': 'instance', 'properties': {'disks': None}}
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        sanitizer.sanitize(body)
        assert {'name': 'instance', 'properties': {'disks': None}} == body

    def test_sanitize_should_not_fail_if_no_specification_matches(self):
        if False:
            i = 10
            return i + 15
        fields_to_sanitize = ['properties.disks.kind1', 'properties.metadata.kind2']
        body = {'name': 'instance', 'properties': {'disks': None}}
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        sanitizer.sanitize(body)
        assert {'name': 'instance', 'properties': {'disks': None}} == body

    def test_sanitize_should_not_fail_if_type_in_body_do_not_match_with_specification(self):
        if False:
            return 10
        fields_to_sanitize = ['properties.disks.kind', 'properties.metadata.kind2']
        body = {'name': 'instance', 'properties': {'disks': 1}}
        sanitizer = GcpBodyFieldSanitizer(fields_to_sanitize)
        sanitizer.sanitize(body)
        assert {'name': 'instance', 'properties': {'disks': 1}} == body