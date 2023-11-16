import pytest
from sentry.models.project import Project
from sentry.testutils.cases import TestCase
from sentry.utils.email import ListResolver
from sentry.utils.email.message_builder import default_list_type_handlers

class ListResolverTestCase(TestCase):
    resolver = ListResolver('namespace', default_list_type_handlers)

    def test_rejects_invalid_namespace(self):
        if False:
            print('Hello World!')
        with pytest.raises(AssertionError):
            ListResolver('\x00', {})

    def test_rejects_invalid_types(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ListResolver.UnregisteredTypeError):
            self.resolver(self.user)

    def test_generates_list_ids(self):
        if False:
            i = 10
            return i + 15
        expected = f'<{self.event.project.slug}.{self.event.organization.slug}.namespace>'
        assert self.resolver(self.event.group) == expected
        assert self.resolver(self.event.project) == expected

    def test_rejects_invalid_objects(self):
        if False:
            return 10
        resolver = ListResolver('namespace', {Project: lambda value: ('\x00',)})
        with pytest.raises(AssertionError):
            resolver(self.project)