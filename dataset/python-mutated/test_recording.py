from __future__ import annotations
import time
import uuid
import zlib
from datetime import datetime
from typing import List, Mapping
from unittest.mock import ANY, patch
import msgpack
from arroyo.backends.kafka import KafkaPayload
from arroyo.types import BrokerValue, Message, Partition, Topic
from sentry_kafka_schemas.schema_types.ingest_replay_recordings_v1 import ReplayRecording
from sentry import options
from sentry.models.files.file import File
from sentry.models.organizationonboardingtask import OnboardingTask, OnboardingTaskStatus
from sentry.replays.consumers.recording import ProcessReplayRecordingStrategyFactory
from sentry.replays.lib.storage import FilestoreBlob, RecordingSegmentStorageMeta, StorageBlob
from sentry.replays.models import ReplayRecordingSegment
from sentry.testutils.abstract import Abstract
from sentry.testutils.cases import TransactionTestCase

def test_multiprocessing_strategy():
    if False:
        return 10
    factory = ProcessReplayRecordingStrategyFactory(num_processes=2, num_threads=1, input_block_size=1, max_batch_size=1, max_batch_time=1, output_block_size=1)

    def _commit(offsets: Mapping[Partition, int], force: bool=False) -> None:
        if False:
            while True:
                i = 10
        return None
    task = factory.create_with_partitions(_commit, {})
    task.terminate()

class RecordingTestCaseMixin(TransactionTestCase):
    __test__ = Abstract(__module__, __qualname__)
    replay_id = uuid.uuid4().hex
    replay_recording_id = uuid.uuid4().hex
    force_synchronous = True

    def assert_replay_recording_segment(self, segment_id: int, compressed: bool) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def processing_factory(self):
        if False:
            for i in range(10):
                print('nop')
        return ProcessReplayRecordingStrategyFactory(input_block_size=1, max_batch_size=1, max_batch_time=1, num_processes=1, num_threads=1, output_block_size=1, force_synchronous=self.force_synchronous)

    def submit(self, messages):
        if False:
            print('Hello World!')
        strategy = self.processing_factory().create_with_partitions(lambda x, force=False: None, None)
        for message in messages:
            strategy.submit(Message(BrokerValue(KafkaPayload(b'key', msgpack.packb(message), [('should_drop', b'1')]), Partition(Topic('ingest-replay-recordings'), 1), 1, datetime.now())))
        strategy.poll()
        strategy.join(1)
        strategy.terminate()

    def nonchunked_messages(self, message: bytes=b'[{"hello":"world"}]', segment_id: int=0, compressed: bool=False) -> List[ReplayRecording]:
        if False:
            while True:
                i = 10
        message = zlib.compress(message) if compressed else message
        return [{'type': 'replay_recording_not_chunked', 'replay_id': self.replay_id, 'org_id': self.organization.id, 'key_id': 123, 'project_id': self.project.id, 'received': int(time.time()), 'retention_days': 30, 'payload': f'{{"segment_id":{segment_id}}}\n'.encode() + message}]

    @patch('sentry.models.OrganizationOnboardingTask.objects.record')
    @patch('sentry.analytics.record')
    def test_compressed_segment_ingestion(self, mock_record, mock_onboarding_task):
        if False:
            i = 10
            return i + 15
        segment_id = 0
        self.submit(self.nonchunked_messages(segment_id=segment_id, compressed=True))
        self.assert_replay_recording_segment(segment_id, compressed=True)
        self.project.refresh_from_db()
        assert self.project.flags.has_replays
        mock_onboarding_task.assert_called_with(organization_id=self.project.organization_id, task=OnboardingTask.SESSION_REPLAY, status=OnboardingTaskStatus.COMPLETE, date_completed=ANY)
        mock_record.assert_called_with('first_replay.sent', organization_id=self.organization.id, project_id=self.project.id, platform=self.project.platform, user_id=self.organization.default_owner_id)

    @patch('sentry.models.OrganizationOnboardingTask.objects.record')
    @patch('sentry.analytics.record')
    def test_uncompressed_segment_ingestion(self, mock_record, mock_onboarding_task):
        if False:
            return 10
        segment_id = 0
        self.submit(self.nonchunked_messages(segment_id=segment_id, compressed=False))
        self.assert_replay_recording_segment(segment_id, False)
        self.project.refresh_from_db()
        assert self.project.flags.has_replays
        mock_onboarding_task.assert_called_with(organization_id=self.project.organization_id, task=OnboardingTask.SESSION_REPLAY, status=OnboardingTaskStatus.COMPLETE, date_completed=ANY)
        mock_record.assert_called_with('first_replay.sent', organization_id=self.organization.id, project_id=self.project.id, platform=self.project.platform, user_id=self.organization.default_owner_id)

class FilestoreRecordingTestCase(RecordingTestCaseMixin):

    def setUp(self):
        if False:
            return 10
        options.set('replay.storage.direct-storage-sample-rate', 0)

    def assert_replay_recording_segment(self, segment_id: int, compressed: bool) -> None:
        if False:
            while True:
                i = 10
        recording_segment = ReplayRecordingSegment.objects.first()
        assert recording_segment.project_id == self.project.id
        assert recording_segment.replay_id == self.replay_id
        assert recording_segment.segment_id == segment_id
        assert recording_segment.file_id is not None
        bytes = self.get_recording_data(segment_id)
        assert len(bytes) == recording_segment.size
        if compressed:
            assert zlib.decompress(bytes) == b'[{"hello":"world"}]'
        else:
            assert bytes == b'[{"hello":"world"}]'

    def get_recording_data(self, segment_id):
        if False:
            return 10
        file = File.objects.first()
        recording_segment = RecordingSegmentStorageMeta(project_id=self.project.id, replay_id=self.replay_id, segment_id=segment_id, retention_days=30, file_id=file.id)
        return FilestoreBlob().get(recording_segment)

class StorageRecordingTestCase(RecordingTestCaseMixin):

    def setUp(self):
        if False:
            return 10
        options.set('replay.storage.direct-storage-sample-rate', 100)

    def assert_replay_recording_segment(self, segment_id: int, compressed: bool) -> None:
        if False:
            i = 10
            return i + 15
        recording_segment = ReplayRecordingSegment.objects.first()
        assert recording_segment is None
        bytes = self.get_recording_data(segment_id)
        if compressed:
            assert zlib.decompress(bytes) == b'[{"hello":"world"}]'
        else:
            assert bytes == b'[{"hello":"world"}]'

    def get_recording_data(self, segment_id):
        if False:
            for i in range(10):
                print('nop')
        recording_segment = RecordingSegmentStorageMeta(project_id=self.project.id, replay_id=self.replay_id, segment_id=segment_id, retention_days=30)
        return StorageBlob().get(recording_segment)

class ThreadedFilestoreRecordingTestCase(FilestoreRecordingTestCase):
    force_synchronous = False

class ThreadedStorageRecordingTestCase(StorageRecordingTestCase):
    force_synchronous = False