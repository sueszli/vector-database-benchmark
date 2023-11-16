import copy
from datetime import timedelta, timezone
from unittest.mock import patch
from urllib.parse import urlencode
import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from sentry.discover.models import DiscoverSavedQuery
from sentry.testutils.cases import AcceptanceTestCase, SnubaTestCase
from sentry.testutils.helpers.datetime import before_now, iso_format, timestamp_format
from sentry.testutils.silo import no_silo_test
from sentry.utils.samples import load_data
FEATURE_NAMES = ['organizations:discover-basic', 'organizations:discover-query', 'organizations:performance-view', 'organizations:performance-tracing-without-performance']

def all_events_query(**kwargs):
    if False:
        while True:
            i = 10
    options = {'sort': ['-timestamp'], 'field': ['title', 'event.type', 'project', 'user.display', 'timestamp'], 'name': ['All Events']}
    options.update(kwargs)
    return urlencode(options, doseq=True)

def errors_query(**kwargs):
    if False:
        i = 10
        return i + 15
    options = {'sort': ['-title'], 'name': ['Errors'], 'field': ['title', 'count(id)', 'count_unique(user)', 'project'], 'query': ['event.type:error']}
    options.update(kwargs)
    return urlencode(options, doseq=True)

def transactions_query(**kwargs):
    if False:
        return 10
    options = {'sort': ['-count'], 'name': ['Transactions'], 'field': ['transaction', 'project', 'count()'], 'statsPeriod': ['14d'], 'query': ['event.type:transaction']}
    options.update(kwargs)
    return urlencode(options, doseq=True)

def transactions_sorted_query(**kwargs):
    if False:
        i = 10
        return i + 15
    options = {'sort': ['transaction'], 'name': ['Transactions'], 'field': ['transaction', 'project', 'count()'], 'statsPeriod': ['14d'], 'query': ['event.type:transaction']}
    options.update(kwargs)
    return urlencode(options, doseq=True)

def generate_transaction(trace=None, span=None):
    if False:
        while True:
            i = 10
    end_datetime = before_now(minutes=10)
    start_datetime = end_datetime - timedelta(milliseconds=500)
    event_data = load_data('transaction', timestamp=end_datetime, start_timestamp=start_datetime, trace=trace, span_id=span)
    event_data.update({'event_id': 'a' * 32})
    reference_span = event_data['spans'][0]
    parent_span_id = reference_span['parent_span_id']
    span_tree_blueprint = {'a': {}, 'b': {'bb': {'bbb': {'bbbb': 'bbbbb'}}}, 'c': {}, 'd': {}, 'e': {}}
    time_offsets = {'a': (timedelta(), timedelta(milliseconds=10)), 'b': (timedelta(milliseconds=120), timedelta(milliseconds=250)), 'bb': (timedelta(milliseconds=130), timedelta(milliseconds=10)), 'bbb': (timedelta(milliseconds=140), timedelta(milliseconds=10)), 'bbbb': (timedelta(milliseconds=150), timedelta(milliseconds=10)), 'bbbbb': (timedelta(milliseconds=160), timedelta(milliseconds=90)), 'c': (timedelta(milliseconds=260), timedelta(milliseconds=100)), 'd': (timedelta(milliseconds=375), timedelta(milliseconds=50)), 'e': (timedelta(milliseconds=400), timedelta(milliseconds=100))}

    def build_span_tree(span_tree, spans, parent_span_id):
        if False:
            for i in range(10):
                print('nop')
        for (span_id, child) in sorted(span_tree.items(), key=lambda item: item[0]):
            span = copy.deepcopy(reference_span)
            span['parent_span_id'] = parent_span_id.ljust(16, '0')
            span['span_id'] = span_id.ljust(16, '0')
            (start_delta, span_length) = time_offsets.get(span_id, (timedelta(), timedelta()))
            span_start_time = start_datetime + start_delta
            span['start_timestamp'] = timestamp_format(span_start_time)
            span['timestamp'] = timestamp_format(span_start_time + span_length)
            spans.append(span)
            if isinstance(child, dict):
                spans = build_span_tree(child, spans, span_id)
            elif isinstance(child, str):
                parent_span_id = span_id
                span_id = child
                span = copy.deepcopy(reference_span)
                span['parent_span_id'] = parent_span_id.ljust(16, '0')
                span['span_id'] = span_id.ljust(16, '0')
                (start_delta, span_length) = time_offsets.get(span_id, (timedelta(), timedelta()))
                span_start_time = start_datetime + start_delta
                span['start_timestamp'] = timestamp_format(span_start_time)
                span['timestamp'] = timestamp_format(span_start_time + span_length)
                spans.append(span)
        return spans
    event_data['spans'] = build_span_tree(span_tree_blueprint, [], parent_span_id)
    return event_data

@no_silo_test(stable=True)
class OrganizationEventsV2Test(AcceptanceTestCase, SnubaTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.user = self.create_user('foo@example.com', is_superuser=True)
        self.org = self.create_organization(name='Rowdy Tiger')
        self.team = self.create_team(organization=self.org, name='Mariachi Band')
        self.project = self.create_project(organization=self.org, teams=[self.team], name='Bengal')
        self.create_member(user=self.user, organization=self.org, role='owner', teams=[self.team])
        self.login_as(self.user)
        self.landing_path = f'/organizations/{self.org.slug}/discover/queries/'
        self.result_path = f'/organizations/{self.org.slug}/discover/results/'

    def wait_until_loaded(self):
        if False:
            i = 10
            return i + 15
        self.browser.wait_until_not('[data-test-id="loading-indicator"]')
        self.browser.wait_until_not('[data-test-id="loading-placeholder"]')

    def test_events_default_landing(self):
        if False:
            for i in range(10):
                print('nop')
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.landing_path)
            self.wait_until_loaded()

    def test_all_events_query_empty_state(self):
        if False:
            return 10
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + all_events_query())
            self.wait_until_loaded()
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + all_events_query(tag=[]))
            self.wait_until_loaded()

    @patch('django.utils.timezone.now')
    def test_all_events_query(self, mock_now):
        if False:
            while True:
                i = 10
        now = before_now().replace(tzinfo=timezone.utc)
        mock_now.return_value = now
        five_mins_ago = iso_format(now - timedelta(minutes=5))
        ten_mins_ago = iso_format(now - timedelta(minutes=10))
        self.store_event(data={'event_id': 'a' * 32, 'message': 'oh no', 'timestamp': five_mins_ago, 'fingerprint': ['group-1']}, project_id=self.project.id, assert_no_errors=False)
        self.store_event(data={'event_id': 'b' * 32, 'message': 'this is bad.', 'timestamp': ten_mins_ago, 'fingerprint': ['group-2'], 'user': {'id': '123', 'email': 'someone@example.com', 'username': 'haveibeenpwned', 'ip_address': '8.8.8.8', 'name': 'Someone'}}, project_id=self.project.id, assert_no_errors=False)
        self.wait_for_event_count(self.project.id, 2)
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + all_events_query())
            self.wait_until_loaded()
            self.browser.wait_until('[data-test-id="grid-editable"] > tbody > tr:nth-child(2)')
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + all_events_query(tag=[]))
            self.wait_until_loaded()
            self.browser.wait_until('[data-test-id="grid-editable"] > tbody > tr:nth-child(2)')

    def test_errors_query_empty_state(self):
        if False:
            i = 10
            return i + 15
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + errors_query())
            self.wait_until_loaded()
            self.browser.click_when_visible('[data-test-id="grid-edit-enable"]')

    @patch('django.utils.timezone.now')
    def test_errors_query(self, mock_now):
        if False:
            i = 10
            return i + 15
        now = before_now().replace(tzinfo=timezone.utc)
        mock_now.return_value = now
        ten_mins_ago = iso_format(now - timedelta(minutes=10))
        self.store_event(data={'event_id': 'a' * 32, 'message': 'oh no', 'timestamp': ten_mins_ago, 'fingerprint': ['group-1'], 'type': 'error'}, project_id=self.project.id, assert_no_errors=False)
        self.store_event(data={'event_id': 'b' * 32, 'message': 'oh no', 'timestamp': ten_mins_ago, 'fingerprint': ['group-1'], 'type': 'error'}, project_id=self.project.id, assert_no_errors=False)
        self.store_event(data={'event_id': 'c' * 32, 'message': 'this is bad.', 'timestamp': ten_mins_ago, 'fingerprint': ['group-2'], 'type': 'error'}, project_id=self.project.id, assert_no_errors=False)
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + errors_query())
            self.wait_until_loaded()

    def test_transactions_query_empty_state(self):
        if False:
            i = 10
            return i + 15
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + transactions_query())
            self.wait_until_loaded()
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + transactions_query(tag=[]))
            self.wait_until_loaded()

    @patch('django.utils.timezone.now')
    def test_transactions_query(self, mock_now):
        if False:
            return 10
        mock_now.return_value = before_now().replace(tzinfo=timezone.utc)
        event_data = generate_transaction()
        self.store_event(data=event_data, project_id=self.project.id, assert_no_errors=True)
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + transactions_query())
            self.wait_until_loaded()
            self.browser.wait_until_not('[data-test-id="grid-editable"] [data-test-id="empty-state"]', timeout=2)

    @patch('django.utils.timezone.now')
    def test_event_detail_view_from_all_events(self, mock_now):
        if False:
            while True:
                i = 10
        now = before_now().replace(tzinfo=timezone.utc)
        mock_now.return_value = now
        ten_mins_ago = iso_format(now - timedelta(minutes=10))
        event_data = load_data('python')
        event_data.update({'event_id': 'a' * 32, 'timestamp': ten_mins_ago, 'received': ten_mins_ago, 'fingerprint': ['group-1']})
        if 'contexts' not in event_data:
            event_data['contexts'] = {}
        event_data['contexts']['trace'] = {'type': 'trace', 'trace_id': 'a' * 32, 'span_id': 'b' * 16}
        self.store_event(data=event_data, project_id=self.project.id, assert_no_errors=False)
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + all_events_query())
            self.wait_until_loaded()
            self.browser.elements('[data-test-id="view-event"]')[0].click()
            self.wait_until_loaded()

    @patch('django.utils.timezone.now')
    def test_event_detail_view_from_errors_view(self, mock_now):
        if False:
            i = 10
            return i + 15
        now = before_now().replace(tzinfo=timezone.utc)
        mock_now.return_value = now
        event_data = load_data('javascript')
        event_data.update({'timestamp': iso_format(now - timedelta(minutes=5)), 'event_id': 'd' * 32, 'fingerprint': ['group-1']})
        event_data['contexts']['trace'] = {'type': 'trace', 'trace_id': 'a' * 32, 'span_id': 'b' * 16}
        self.store_event(data=event_data, project_id=self.project.id)
        self.wait_for_event_count(self.project.id, 1)
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + errors_query() + '&statsPeriod=24h')
            self.wait_until_loaded()
            self.browser.element('[data-test-id="open-group"]').click()
            self.wait_until_loaded()
            self.browser.elements('[data-test-id="view-event"]')[0].click()
            self.wait_until_loaded()

    @patch('django.utils.timezone.now')
    def test_event_detail_view_from_transactions_query(self, mock_now):
        if False:
            print('Hello World!')
        mock_now.return_value = before_now().replace(tzinfo=timezone.utc)
        event_data = generate_transaction(trace='a' * 32, span='ab' * 8)
        self.store_event(data=event_data, project_id=self.project.id, assert_no_errors=True)
        child_event = generate_transaction(trace=event_data['contexts']['trace']['trace_id'], span='bc' * 8)
        child_event['event_id'] = 'b' * 32
        child_event['contexts']['trace']['parent_span_id'] = event_data['spans'][4]['span_id']
        child_event['transaction'] = 'z-child-transaction'
        child_event['spans'] = child_event['spans'][0:3]
        self.store_event(data=child_event, project_id=self.project.id, assert_no_errors=True)
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + transactions_sorted_query())
            self.wait_until_loaded()
            self.browser.elements('[data-test-id="open-group"]')[0].click()
            self.wait_until_loaded()
            self.browser.elements('[data-test-id="view-event"]')[0].click()
            self.wait_until_loaded()
            self.browser.element('[data-test-id="span-row-5"]').click()
            self.browser.element('[data-test-id="span-row-7"]').click()
            child_button = '[data-test-id="view-child-transaction"]'
            self.browser.wait_until(child_button)
            self.browser.click(child_button)
            self.wait_until_loaded()

    @patch('django.utils.timezone.now')
    def test_event_detail_view_from_transactions_query_siblings(self, mock_now):
        if False:
            i = 10
            return i + 15
        mock_now.return_value = before_now().replace(tzinfo=timezone.utc)
        event_data = generate_transaction(trace='a' * 32, span='ab' * 8)
        last_span = copy.deepcopy(event_data['spans'][-1])
        for i in range(5):
            clone = copy.deepcopy(last_span)
            clone['span_id'] = (str('ac' * 6) + str(i)).ljust(16, '0')
            event_data['spans'].append(clone)
        combo_breaker_span = copy.deepcopy(last_span)
        combo_breaker_span['span_id'] = str('af' * 6).ljust(16, '0')
        combo_breaker_span['op'] = 'combo.breaker'
        event_data['spans'].append(combo_breaker_span)
        for i in range(5):
            clone = copy.deepcopy(last_span)
            clone['op'] = 'django.middleware'
            clone['span_id'] = (str('de' * 6) + str(i)).ljust(16, '0')
            event_data['spans'].append(clone)
        for i in range(5):
            clone = copy.deepcopy(last_span)
            clone['op'] = 'http'
            clone['description'] = 'test'
            clone['span_id'] = (str('bd' * 6) + str(i)).ljust(16, '0')
            event_data['spans'].append(clone)
        self.store_event(data=event_data, project_id=self.project.id, assert_no_errors=True)
        child_event = generate_transaction(trace=event_data['contexts']['trace']['trace_id'], span='bc' * 8)
        child_event['event_id'] = 'b' * 32
        child_event['contexts']['trace']['parent_span_id'] = event_data['spans'][4]['span_id']
        child_event['transaction'] = 'z-child-transaction'
        child_event['spans'] = child_event['spans'][0:3]
        self.store_event(data=child_event, project_id=self.project.id, assert_no_errors=True)
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + transactions_sorted_query())
            self.wait_until_loaded()
            self.browser.elements('[data-test-id="open-group"]')[0].click()
            self.wait_until_loaded()
            self.browser.elements('[data-test-id="view-event"]')[0].click()
            self.wait_until_loaded()
            self.browser.element('[data-test-id="span-row-5"]').click()
            self.browser.element('[data-test-id="span-row-9"]').click()
            self.browser.element('[data-test-id="span-row-18"]').click()
            self.browser.element('[data-test-id="span-row-23"]').click()
            first_row = self.browser.element('[data-test-id="span-row-23"]')
            first_row.find_element(By.CSS_SELECTOR, 'a').click()
            second_row = self.browser.element('[data-test-id="span-row-18"]')
            second_row.find_element(By.CSS_SELECTOR, 'a').click()
            third_row = self.browser.element('[data-test-id="span-row-9"]')
            third_row.find_element(By.CSS_SELECTOR, 'a').click()

    @patch('django.utils.timezone.now')
    def test_transaction_event_detail_view_ops_filtering(self, mock_now):
        if False:
            i = 10
            return i + 15
        mock_now.return_value = before_now().replace(tzinfo=timezone.utc)
        event_data = generate_transaction(trace='a' * 32, span='ab' * 8)
        self.store_event(data=event_data, project_id=self.project.id, assert_no_errors=True)
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + transactions_query())
            self.wait_until_loaded()
            self.browser.elements('[data-test-id="open-group"]')[0].click()
            self.wait_until_loaded()
            self.browser.elements('[data-test-id="view-event"]')[0].click()
            self.wait_until_loaded()
            self.browser.elements('[aria-label="Filter by operation"]')[0].click()
            self.browser.elements('[data-test-id="django\\\\.middleware"]')[0].click()

    def test_create_saved_query(self):
        if False:
            print('Hello World!')
        query = {'field': ['project.id', 'count()'], 'query': 'event.type:error'}
        query_name = 'A new custom query'
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + urlencode(query, doseq=True))
            self.wait_until_loaded()
            self.browser.element('[aria-label="Save as"]').click()
            self.browser.element('input[name="query_name"]').send_keys(query_name)
            self.browser.element('[aria-label="Save for Org"]').click()
            self.browser.wait_until(f'[data-test-id="discover2-query-name-{query_name}"]')
            editable_text_label = self.browser.element('[data-test-id="editable-text-label"]').text
        assert editable_text_label == query_name
        assert DiscoverSavedQuery.objects.filter(name=query_name).exists()

    def test_view_and_rename_saved_query(self):
        if False:
            return 10
        query = DiscoverSavedQuery.objects.create(name='Custom query', organization=self.org, version=2, query={'fields': ['title', 'project.id', 'count()'], 'query': 'event.type:error'})
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.landing_path)
            self.wait_until_loaded()
            self.browser.element(f'[data-test-id="card-{query.name}"]').click()
            self.wait_until_loaded()
            self.browser.element('[data-test-id="editable-text-label"]').click()
            self.browser.wait_until('[data-test-id="editable-text-input"]')
            editable_text_input = self.browser.element('[data-test-id="editable-text-input"] input')
            editable_text_input.click()
            editable_text_input.send_keys(Keys.END + 'updated!')
            self.browser.element('table').click()
            self.browser.wait_until('[data-test-id="editable-text-label"]')
            new_name = 'Custom queryupdated!'
            self.browser.wait_until(f'[data-test-id="discover2-query-name-{new_name}"]')
        assert DiscoverSavedQuery.objects.filter(name=new_name).exists()

    def test_delete_saved_query(self):
        if False:
            while True:
                i = 10
        query = DiscoverSavedQuery.objects.create(name='Custom query', organization=self.org, version=2, query={'fields': ['title', 'project.id', 'count()'], 'query': 'event.type:error'})
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.landing_path)
            self.wait_until_loaded()
            card_selector = f'[data-test-id="card-{query.name}"]'
            card = self.browser.element(card_selector)
            card.find_element(by=By.CSS_SELECTOR, value='[data-test-id="menu-trigger"]').click()
            card.find_element(by=By.CSS_SELECTOR, value='[data-test-id="delete"]').click()
            self.browser.wait_until_not(card_selector)
            assert DiscoverSavedQuery.objects.filter(name=query.name).exists() is False

    def test_duplicate_query(self):
        if False:
            print('Hello World!')
        query = DiscoverSavedQuery.objects.create(name='Custom query', organization=self.org, version=2, query={'fields': ['title', 'project.id', 'count()'], 'query': 'event.type:error'})
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.landing_path)
            self.wait_until_loaded()
            card_selector = f'[data-test-id="card-{query.name}"]'
            card = self.browser.element(card_selector)
            card.find_element(by=By.CSS_SELECTOR, value='[data-test-id="menu-trigger"]').click()
            card.find_element(by=By.CSS_SELECTOR, value='[data-test-id="duplicate"]').click()
            duplicate_name = f'{query.name} copy'
            self.browser.get(self.landing_path)
            self.browser.element(f'[data-test-id="card-{duplicate_name}"]')
            assert DiscoverSavedQuery.objects.filter(name=duplicate_name).exists()

    @pytest.mark.skip(reason='causing timeouts in github actions and travis')
    @patch('django.utils.timezone.now')
    def test_drilldown_result(self, mock_now):
        if False:
            i = 10
            return i + 15
        now = before_now().replace(tzinfo=timezone.utc)
        mock_now.return_value = now
        ten_mins_ago = iso_format(now - timedelta(minutes=10))
        events = (('a' * 32, 'oh no', 'group-1'), ('b' * 32, 'oh no', 'group-1'), ('c' * 32, 'this is bad', 'group-2'))
        for event in events:
            self.store_event(data={'event_id': event[0], 'message': event[1], 'timestamp': ten_mins_ago, 'fingerprint': [event[2]], 'type': 'error'}, project_id=self.project.id)
        query = {'field': ['message', 'project', 'count()'], 'query': 'event.type:error'}
        with self.feature(FEATURE_NAMES):
            self.browser.get(self.result_path + '?' + urlencode(query, doseq=True))
            self.wait_until_loaded()
            self.browser.element('[data-test-id="expand-count"]').click()
            self.wait_until_loaded()
            assert self.browser.element_exists_by_test_id('grid-editable'), 'table should exist.'
            headers = self.browser.elements('[data-test-id="grid-editable"] thead th')
            expected = ['', 'MESSAGE', 'PROJECT', 'ID']
            actual = [header.text for header in headers]
            assert expected == actual

    @pytest.mark.skip(reason='not done')
    @patch('django.utils.timezone.now')
    def test_usage(self, mock_now):
        if False:
            for i in range(10):
                print('nop')
        mock_now.return_value = before_now().replace(tzinfo=timezone.utc)