from os import environ
from os.path import basename
import threading
import time
from typing import Type
import uuid
from google.api_core.exceptions import AlreadyExists
from google.api_core.exceptions import InvalidArgument
from google.api_core.exceptions import NotFound
from google.cloud.devtools import containeranalysis_v1
from google.cloud.pubsub import PublisherClient, SubscriberClient
from google.cloud.pubsub_v1.subscriber.message import Message
from grafeas.grafeas_v1 import DiscoveryOccurrence
from grafeas.grafeas_v1 import NoteKind
from grafeas.grafeas_v1 import Severity
from grafeas.grafeas_v1 import Version
import pytest
from create_note import create_note
from create_occurrence import create_occurrence
from create_occurrence_subscription import create_occurrence_subscription
from delete_note import delete_note
from delete_occurrence import delete_occurrence
from find_high_severity_vulnerabilities_for_image import find_high_severity_vulnerabilities_for_image
from find_vulnerabilities_for_image import find_vulnerabilities_for_image
from get_note import get_note
from get_occurrence import get_occurrence
from get_occurrences_for_image import get_occurrences_for_image
from get_occurrences_for_note import get_occurrences_for_note
from poll_discovery_finished import poll_discovery_finished
PROJECT_ID = environ['GOOGLE_CLOUD_PROJECT']
SLEEP_TIME = 1
TRY_LIMIT = 20

class MessageReceiver:
    """Custom class to handle incoming Pub/Sub messages."""

    def __init__(self, expected_msg_nums: int, done_event: threading.Event) -> None:
        if False:
            return 10
        self.msg_count = 0
        self.expected_msg_nums = expected_msg_nums
        self.done_event = done_event

    def pubsub_callback(self, message: Message) -> None:
        if False:
            while True:
                i = 10
        self.msg_count += 1
        print(f'Message {self.msg_count}: {message.data}')
        message.ack()
        if self.msg_count == self.expected_msg_nums:
            self.done_event.set()

class TestContainerAnalysisSamples:

    def setup_method(self, test_method: Type[MessageReceiver]) -> None:
        if False:
            while True:
                i = 10
        print(f'SETUP {test_method.__name__}')
        self.note_id = f'note-{uuid.uuid4()}'
        self.image_url = f'{uuid.uuid4()}.{test_method.__name__}'
        self.note_obj = create_note(self.note_id, PROJECT_ID)

    def teardown_method(self, test_method: Type[MessageReceiver]) -> None:
        if False:
            return 10
        print(f'TEAR DOWN {test_method.__name__}')
        try:
            delete_note(self.note_id, PROJECT_ID)
        except NotFound:
            pass

    def test_create_note(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        new_note = get_note(self.note_id, PROJECT_ID)
        assert new_note.name == self.note_obj.name

    def test_delete_note(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        delete_note(self.note_id, PROJECT_ID)
        try:
            get_note(self.note_obj, PROJECT_ID)
        except InvalidArgument:
            pass
        else:
            assert False

    def test_create_occurrence(self) -> None:
        if False:
            return 10
        created = create_occurrence(self.image_url, self.note_id, PROJECT_ID, PROJECT_ID)
        retrieved = get_occurrence(basename(created.name), PROJECT_ID)
        assert created.name == retrieved.name
        delete_occurrence(basename(created.name), PROJECT_ID)

    def test_delete_occurrence(self) -> None:
        if False:
            return 10
        created = create_occurrence(self.image_url, self.note_id, PROJECT_ID, PROJECT_ID)
        delete_occurrence(basename(created.name), PROJECT_ID)
        try:
            get_occurrence(basename(created.name), PROJECT_ID)
        except NotFound:
            pass
        else:
            assert False

    def test_occurrences_for_image(self) -> None:
        if False:
            i = 10
            return i + 15
        orig_count = get_occurrences_for_image(self.image_url, PROJECT_ID)
        occ = create_occurrence(self.image_url, self.note_id, PROJECT_ID, PROJECT_ID)
        new_count = 0
        tries = 0
        while new_count != 1 and tries < TRY_LIMIT:
            tries += 1
            new_count = get_occurrences_for_image(self.image_url, PROJECT_ID)
            time.sleep(SLEEP_TIME)
        assert new_count == 1
        assert orig_count == 0
        delete_occurrence(basename(occ.name), PROJECT_ID)

    def test_occurrences_for_note(self) -> None:
        if False:
            while True:
                i = 10
        orig_count = get_occurrences_for_note(self.note_id, PROJECT_ID)
        occ = create_occurrence(self.image_url, self.note_id, PROJECT_ID, PROJECT_ID)
        new_count = 0
        tries = 0
        while new_count != 1 and tries < TRY_LIMIT:
            tries += 1
            new_count = get_occurrences_for_note(self.note_id, PROJECT_ID)
            time.sleep(SLEEP_TIME)
        assert new_count == 1
        assert orig_count == 0
        delete_occurrence(basename(occ.name), PROJECT_ID)

    @pytest.mark.flaky(max_runs=3, min_passes=1)
    def test_pubsub(self) -> None:
        if False:
            while True:
                i = 10
        client = SubscriberClient()
        try:
            topic_id = 'container-analysis-occurrences-v1'
            topic_name = {'name': f'projects/{PROJECT_ID}/topics/{topic_id}'}
            publisher = PublisherClient()
            publisher.create_topic(topic_name)
        except AlreadyExists:
            pass
        subscription_id = f'container-analysis-test-{uuid.uuid4()}'
        subscription_name = client.subscription_path(PROJECT_ID, subscription_id)
        create_occurrence_subscription(subscription_id, PROJECT_ID)
        message_count = 1
        try:
            job_done = threading.Event()
            receiver = MessageReceiver(message_count, job_done)
            client.subscribe(subscription_name, receiver.pubsub_callback)
            for i in range(message_count):
                occ = create_occurrence(self.image_url, self.note_id, PROJECT_ID, PROJECT_ID)
                time.sleep(SLEEP_TIME)
                delete_occurrence(basename(occ.name), PROJECT_ID)
                time.sleep(SLEEP_TIME)
            job_done.wait(timeout=180)
            print(f'done. msg_count = {receiver.msg_count}')
            assert message_count <= receiver.msg_count
        finally:
            client.delete_subscription({'subscription': subscription_name})

    def test_poll_discovery_occurrence_fails(self) -> None:
        if False:
            i = 10
            return i + 15
        try:
            poll_discovery_finished(self.image_url, 5, PROJECT_ID)
        except RuntimeError:
            pass
        else:
            assert False

    @pytest.mark.flaky(max_runs=3, min_passes=1)
    def test_poll_discovery_occurrence(self) -> None:
        if False:
            return 10
        note_id = f'discovery-note-{uuid.uuid4()}'
        client = containeranalysis_v1.ContainerAnalysisClient()
        grafeas_client = client.get_grafeas_client()
        note = {'discovery': {'analysis_kind': NoteKind.DISCOVERY}}
        grafeas_client.create_note(parent=f'projects/{PROJECT_ID}', note_id=note_id, note=note)
        occurrence = {'note_name': f'projects/{PROJECT_ID}/notes/{note_id}', 'resource_uri': self.image_url, 'discovery': {'analysis_status': DiscoveryOccurrence.AnalysisStatus.FINISHED_SUCCESS}}
        created = grafeas_client.create_occurrence(parent=f'projects/{PROJECT_ID}', occurrence=occurrence)
        disc = poll_discovery_finished(self.image_url, 10, PROJECT_ID)
        status = disc.discovery.analysis_status
        assert disc is not None
        assert status == DiscoveryOccurrence.AnalysisStatus.FINISHED_SUCCESS
        delete_occurrence(basename(created.name), PROJECT_ID)
        delete_note(note_id, PROJECT_ID)

    def test_find_vulnerabilities_for_image(self) -> None:
        if False:
            while True:
                i = 10
        occ_list = find_vulnerabilities_for_image(self.image_url, PROJECT_ID)
        assert len(occ_list) == 0
        created = create_occurrence(self.image_url, self.note_id, PROJECT_ID, PROJECT_ID)
        tries = 0
        count = 0
        while count != 1 and tries < TRY_LIMIT:
            tries += 1
            occ_list = find_vulnerabilities_for_image(self.image_url, PROJECT_ID)
            count = len(occ_list)
            time.sleep(SLEEP_TIME)
        assert len(occ_list) == 1
        delete_occurrence(basename(created.name), PROJECT_ID)

    def test_find_high_severity_vulnerabilities(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        occ_list = find_high_severity_vulnerabilities_for_image(self.image_url, PROJECT_ID)
        assert len(occ_list) == 0
        note_id = f'discovery-note-{uuid.uuid4()}'
        client = containeranalysis_v1.ContainerAnalysisClient()
        grafeas_client = client.get_grafeas_client()
        note = {'vulnerability': {'severity': Severity.CRITICAL, 'details': [{'affected_cpe_uri': 'your-uri-here', 'affected_package': 'your-package-here', 'affected_version_start': {'kind': Version.VersionKind.MINIMUM}, 'fixed_version': {'kind': Version.VersionKind.MAXIMUM}}]}}
        grafeas_client.create_note(parent=f'projects/{PROJECT_ID}', note_id=note_id, note=note)
        occurrence = {'note_name': f'projects/{PROJECT_ID}/notes/{note_id}', 'resource_uri': self.image_url, 'vulnerability': {'effective_severity': Severity.CRITICAL, 'package_issue': [{'affected_cpe_uri': 'your-uri-here', 'affected_package': 'your-package-here', 'affected_version': {'kind': Version.VersionKind.MINIMUM}, 'fixed_version': {'kind': Version.VersionKind.MAXIMUM}}]}}
        created = grafeas_client.create_occurrence(parent=f'projects/{PROJECT_ID}', occurrence=occurrence)
        tries = 0
        count = 0
        while count != 1 and tries < TRY_LIMIT:
            tries += 1
            occ_list = find_vulnerabilities_for_image(self.image_url, PROJECT_ID)
            count = len(occ_list)
            time.sleep(SLEEP_TIME)
        assert len(occ_list) == 1
        delete_occurrence(basename(created.name), PROJECT_ID)
        delete_note(note_id, PROJECT_ID)