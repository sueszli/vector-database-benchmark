import datetime
from io import BytesIO
from unittest import mock
from uuid import uuid4
import pytest
from django.urls import reverse
from sentry.models.files.file import File
from sentry.replays.lib import kafka
from sentry.replays.models import ReplayRecordingSegment
from sentry.replays.testutils import assert_expected_response, mock_expected_response, mock_replay
from sentry.testutils.cases import APITestCase, ReplaysSnubaTestCase
from sentry.testutils.helpers import TaskRunner
from sentry.testutils.silo import region_silo_test
from sentry.utils import kafka_config
REPLAYS_FEATURES = {'organizations:session-replay': True}

@pytest.fixture(autouse=True)
def setup():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(kafka_config, 'get_kafka_producer_cluster_options'):
        with mock.patch.object(kafka, 'KafkaPublisher'):
            yield

@region_silo_test(stable=True)
class ProjectReplayDetailsTest(APITestCase, ReplaysSnubaTestCase):
    endpoint = 'sentry-api-0-project-replay-details'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.login_as(user=self.user)
        self.replay_id = uuid4().hex
        self.url = reverse(self.endpoint, args=(self.organization.slug, self.project.slug, self.replay_id))

    def test_feature_flag_disabled(self):
        if False:
            return 10
        response = self.client.get(self.url)
        assert response.status_code == 404

    def test_no_replay_found(self):
        if False:
            return 10
        with self.feature(REPLAYS_FEATURES):
            response = self.client.get(self.url)
            assert response.status_code == 404

    def test_get_one_replay(self):
        if False:
            while True:
                i = 10
        'Test only one replay returned.'
        replay1_id = self.replay_id
        replay2_id = uuid4().hex
        seq1_timestamp = datetime.datetime.now() - datetime.timedelta(seconds=10)
        seq2_timestamp = datetime.datetime.now() - datetime.timedelta(seconds=5)
        self.store_replays(mock_replay(seq1_timestamp, self.project.id, replay1_id))
        self.store_replays(mock_replay(seq2_timestamp, self.project.id, replay1_id))
        self.store_replays(mock_replay(seq1_timestamp, self.project.id, replay2_id))
        self.store_replays(mock_replay(seq2_timestamp, self.project.id, replay2_id))
        with self.feature(REPLAYS_FEATURES):
            response = self.client.get(self.url)
            assert response.status_code == 200
            response_data = response.json()
            assert 'data' in response_data
            assert response_data['data']['id'] == replay1_id
            response = self.client.get(reverse(self.endpoint, args=(self.organization.slug, self.project.slug, replay2_id)))
            assert response.status_code == 200
            response_data = response.json()
            assert 'data' in response_data
            assert response_data['data']['id'] == replay2_id

    def test_get_replay_schema(self):
        if False:
            print('Hello World!')
        'Test replay schema is well-formed.'
        replay1_id = self.replay_id
        replay2_id = uuid4().hex
        seq1_timestamp = datetime.datetime.now() - datetime.timedelta(seconds=25)
        seq2_timestamp = datetime.datetime.now() - datetime.timedelta(seconds=7)
        seq3_timestamp = datetime.datetime.now() - datetime.timedelta(seconds=4)
        trace_id_1 = uuid4().hex
        trace_id_2 = uuid4().hex
        self.store_replays(mock_replay(seq1_timestamp, self.project.id, replay2_id))
        self.store_replays(mock_replay(seq3_timestamp, self.project.id, replay2_id))
        self.store_replays(mock_replay(seq1_timestamp, self.project.id, replay1_id, trace_ids=[trace_id_1], urls=['http://localhost:3000/']))
        self.store_replays(mock_replay(seq2_timestamp, self.project.id, replay1_id, segment_id=1, trace_ids=[trace_id_2], urls=['http://www.sentry.io/'], error_ids=[]))
        self.store_replays(mock_replay(seq3_timestamp, self.project.id, replay1_id, segment_id=2, trace_ids=[trace_id_2], urls=['http://localhost:3000/'], error_ids=[]))
        with self.feature(REPLAYS_FEATURES):
            response = self.client.get(self.url)
            assert response.status_code == 200
            response_data = response.json()
            assert 'data' in response_data
            expected_response = mock_expected_response(self.project.id, replay1_id, seq1_timestamp, seq3_timestamp, trace_ids=[trace_id_1, trace_id_2], urls=['http://localhost:3000/', 'http://www.sentry.io/', 'http://localhost:3000/'], count_segments=3, activity=4)
            assert_expected_response(response_data['data'], expected_response)

    def test_delete(self):
        if False:
            i = 10
            return i + 15
        user = self.create_user(is_superuser=False)
        self.create_member(user=user, organization=self.organization, role='member', teams=[])
        self.login_as(user=user)
        file = File.objects.create(name='recording-segment-0', type='application/octet-stream')
        file.putfile(BytesIO(b'replay-recording-segment'))
        recording_segment = ReplayRecordingSegment.objects.create(replay_id=self.replay_id, project_id=self.project.id, segment_id=0, file_id=file.id)
        file_id = file.id
        recording_segment_id = recording_segment.id
        with self.feature(REPLAYS_FEATURES):
            with TaskRunner():
                response = self.client.delete(self.url)
                assert response.status_code == 202
        try:
            ReplayRecordingSegment.objects.get(id=recording_segment_id)
            assert False, 'Recording Segment was not deleted.'
        except ReplayRecordingSegment.DoesNotExist:
            pass
        try:
            File.objects.get(id=file_id)
            assert False, 'File was not deleted.'
        except File.DoesNotExist:
            pass