from rest_framework import status
from sentry.models.eventerror import EventError
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test
from sentry.testutils.skips import requires_snuba
pytestmark = [requires_snuba]

@region_silo_test
class ActionableItemsEndpointTestCase(APITestCase):
    endpoint = 'sentry-api-0-event-actionable-items'

    def setUp(self) -> None:
        if False:
            return 10
        self.login_as(self.user)
        return super().setUp()

    def test_missing_event(self):
        if False:
            print('Hello World!')
        resp = self.get_error_response(self.organization.slug, self.project.slug, 'invalid_id', status_code=status.HTTP_404_NOT_FOUND)
        assert resp.data['detail'] == 'Event not found'

    def test_orders_event_errors_by_priority(self):
        if False:
            for i in range(10):
                print('nop')
        event = self.store_event(data={'event_id': 'a' * 32, 'release': 'my-release', 'dist': 'my-dist', 'sdk': {'name': 'sentry.javascript.browser', 'version': '7.3.0'}, 'exception': {'values': [{'type': 'Error', 'stacktrace': {'frames': [{'abs_path': 'https://example.com/application.js', 'lineno': 1, 'colno': 39}]}}]}, 'errors': [{'type': EventError.INVALID_DATA, 'name': 'foo'}, {'type': EventError.JS_MISSING_SOURCES_CONTENT, 'url': 'http://example.com'}, {'type': EventError.UNKNOWN_ERROR, 'name': 'bar'}]}, project_id=self.project.id, assert_no_errors=False)
        resp = self.get_success_response(self.organization.slug, self.project.slug, event.event_id)
        errors = resp.data['errors']
        assert len(errors) == 2
        missing_error = errors[0]
        invalid_data = errors[1]
        assert missing_error['type'] == EventError.JS_MISSING_SOURCES_CONTENT
        assert invalid_data['type'] == EventError.INVALID_DATA