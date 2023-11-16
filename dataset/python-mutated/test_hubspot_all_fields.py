import datetime
import tap_tester.connections as connections
import tap_tester.menagerie as menagerie
import tap_tester.runner as runner
from tap_tester import LOGGER
from base import HubspotBaseTest
from client import TestClient

def get_matching_actual_record_by_pk(expected_primary_key_dict, actual_records):
    if False:
        return 10
    ret_records = []
    can_save = True
    for record in actual_records:
        for (key, value) in expected_primary_key_dict.items():
            actual_value = record[key]
            if actual_value != value:
                can_save = False
                break
        if can_save:
            ret_records.append(record)
        can_save = True
    return ret_records
FIELDS_ADDED_BY_TAP = {'contacts': {'versionTimestamp'}}
KNOWN_EXTRA_FIELDS = {'deals': {'property_hs_date_entered_1258834'}}
KNOWN_MISSING_FIELDS = {'contacts': {'property_hs_latest_source_data_2', 'property_hs_latest_source', 'property_hs_latest_source_data_1', 'property_hs_timezone', 'property_hs_latest_source_timestamp'}, 'contact_lists': {'authorId', 'teamIds', 'internal', 'ilsFilterBranch', 'limitExempt'}, 'email_events': {'portalSubscriptionStatus', 'attempt', 'source', 'subscriptions', 'sourceId', 'replyTo', 'suppressedMessage', 'bcc', 'suppressedReason', 'cc'}, 'workflows': {'migrationStatus', 'updateSource', 'description', 'originalAuthorUserId', 'lastUpdatedByUserId', 'creationSource', 'portalId', 'contactCounts'}, 'owners': {'activeSalesforceId'}, 'forms': {'alwaysCreateNewCompany', 'themeColor', 'publishAt', 'editVersion', 'themeName', 'style', 'thankYouMessageJson', 'createMarketableContact', 'kickbackEmailWorkflowId', 'businessUnitId', 'portableKey', 'parentId', 'kickbackEmailsJson', 'unpublishAt', 'internalUpdatedAt', 'multivariateTest', 'publishedAt', 'customUid', 'isPublished', 'paymentSessionTemplateIds', 'selectedExternalOptions'}, 'companies': {'mergeAudits', 'stateChanges', 'isDeleted', 'additionalDomains', 'property_hs_analytics_latest_source', 'property_hs_analytics_latest_source_data_2', 'property_hs_analytics_latest_source_data_1', 'property_hs_analytics_latest_source_timestamp'}, 'campaigns': {'lastProcessingStateChangeAt', 'lastProcessingFinishedAt', 'processingState', 'lastProcessingStartedAt'}, 'deals': {'imports', 'property_hs_num_associated_deal_splits', 'property_hs_is_deal_split', 'stateChanges', 'property_hs_num_associated_active_deal_registrations', 'property_hs_num_associated_deal_registrations', 'property_hs_analytics_latest_source', 'property_hs_analytics_latest_source_timestamp_contact', 'property_hs_analytics_latest_source_data_1_contact', 'property_hs_analytics_latest_source_timestamp', 'property_hs_analytics_latest_source_data_1', 'property_hs_analytics_latest_source_contact', 'property_hs_analytics_latest_source_company', 'property_hs_analytics_latest_source_data_1_company', 'property_hs_analytics_latest_source_data_2_company', 'property_hs_analytics_latest_source_data_2', 'property_hs_analytics_latest_source_data_2_contact'}, 'subscription_changes': {'normalizedEmailId'}}

class TestHubspotAllFields(HubspotBaseTest):
    """Test that with all fields selected for a stream we replicate data as expected"""

    @staticmethod
    def name():
        if False:
            i = 10
            return i + 15
        return 'tt_hubspot_all_fields_dynamic'

    def streams_under_test(self):
        if False:
            return 10
        'expected streams minus the streams not under test'
        return self.expected_streams().difference({'owners', 'subscription_changes'})

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.maxDiff = None
        test_client = TestClient(start_date=self.get_properties()['start_date'])
        self.expected_records = dict()
        streams = self.streams_under_test()
        stream_to_run_last = 'contacts_by_company'
        if stream_to_run_last in streams:
            streams.remove(stream_to_run_last)
            streams = list(streams)
            streams.append(stream_to_run_last)
        for stream in streams:
            if stream == 'contacts_by_company':
                company_ids = [company['companyId'] for company in self.expected_records['companies']]
                self.expected_records[stream] = test_client.read(stream, parent_ids=company_ids)
            else:
                self.expected_records[stream] = test_client.read(stream)
        for (stream, records) in self.expected_records.items():
            LOGGER.info('The test client found %s %s records.', len(records), stream)
        self.convert_datatype(self.expected_records)

    def convert_datatype(self, expected_records):
        if False:
            print('Hello World!')
        for (stream, records) in expected_records.items():
            for record in records:
                timestamp_keys = {'timestamp'}
                for key in timestamp_keys:
                    timestamp = record.get(key)
                    if timestamp:
                        unformatted = datetime.datetime.fromtimestamp(timestamp / 1000)
                        formatted = datetime.datetime.strftime(unformatted, self.BASIC_DATE_FORMAT)
                        record[key] = formatted
        return expected_records

    def test_run(self):
        if False:
            return 10
        conn_id = connections.ensure_connection(self)
        found_catalogs = self.run_and_verify_check_mode(conn_id)
        expected_streams = self.streams_under_test()
        catalog_entries = [ce for ce in found_catalogs if ce['tap_stream_id'] in expected_streams]
        for catalog_entry in catalog_entries:
            stream_schema = menagerie.get_annotated_schema(conn_id, catalog_entry['stream_id'])
            connections.select_catalog_and_fields_via_metadata(conn_id, catalog_entry, stream_schema)
        first_record_count_by_stream = self.run_and_verify_sync(conn_id)
        synced_records = runner.get_records_from_target_output()
        for stream in expected_streams:
            with self.subTest(stream=stream):
                replication_method = self.expected_replication_method()[stream]
                primary_keys = sorted(self.expected_primary_keys()[stream])
                actual_records = [message['data'] for message in synced_records[stream]['messages'] if message['action'] == 'upsert']
                for expected_record in self.expected_records[stream]:
                    primary_key_dict = {primary_key: expected_record[primary_key] for primary_key in primary_keys}
                    primary_key_values = list(primary_key_dict.values())
                    with self.subTest(expected_record=primary_key_dict):
                        matching_actual_records_by_pk = get_matching_actual_record_by_pk(primary_key_dict, actual_records)
                        if not matching_actual_records_by_pk:
                            LOGGER.warn('Expected %s record was not replicated: %s', stream, primary_key_dict)
                            continue
                        actual_record = matching_actual_records_by_pk[0]
                        expected_keys = set(expected_record.keys()).union(FIELDS_ADDED_BY_TAP.get(stream, {}))
                        actual_keys = set(actual_record.keys())
                        known_missing_keys = set()
                        for missing_key in KNOWN_MISSING_FIELDS.get(stream, set()):
                            if missing_key in expected_record.keys():
                                known_missing_keys.add(missing_key)
                                del expected_record[missing_key]
                        known_extra_keys = set()
                        for extra_key in KNOWN_EXTRA_FIELDS.get(stream, set()):
                            known_extra_keys.add(extra_key)
                        expected_keys_adjusted = expected_keys.union(known_extra_keys)
                        actual_keys_adjusted = actual_keys.union(known_missing_keys)
                        bad_key_prefixes = {'property_hs_date_entered_', 'property_hs_date_exited_'}
                        bad_keys = set()
                        for key in expected_keys_adjusted:
                            for prefix in bad_key_prefixes:
                                if key.startswith(prefix) and key not in actual_keys_adjusted:
                                    bad_keys.add(key)
                        for key in actual_keys_adjusted:
                            for prefix in bad_key_prefixes:
                                if key.startswith(prefix) and key not in expected_keys_adjusted:
                                    bad_keys.add(key)
                        for key in bad_keys:
                            if key in expected_keys_adjusted:
                                expected_keys_adjusted.remove(key)
                            elif key in actual_keys_adjusted:
                                actual_keys_adjusted.remove(key)
                        self.assertSetEqual(expected_keys_adjusted, actual_keys_adjusted)
                expected_primary_key_values = {tuple([record[primary_key] for primary_key in primary_keys]) for record in self.expected_records[stream]}
                actual_records_primary_key_values = {tuple([record[primary_key] for primary_key in primary_keys]) for record in actual_records}
                if expected_primary_key_values.issubset(actual_records_primary_key_values):
                    LOGGER.warn('Unexpected %s records replicated: %s', stream, actual_records_primary_key_values - expected_primary_key_values)

class TestHubspotAllFieldsStatic(TestHubspotAllFields):

    @staticmethod
    def name():
        if False:
            i = 10
            return i + 15
        return 'tt_hubspot_all_fields_static'

    def streams_under_test(self):
        if False:
            i = 10
            return i + 15
        'expected streams minus the streams not under test'
        return {'owners'}

    def get_properties(self):
        if False:
            print('Hello World!')
        return {'start_date': '2021-05-02T00:00:00Z'}