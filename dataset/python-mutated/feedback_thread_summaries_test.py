"""Various load tests which ensure that the time for a particular process
is within a given limit.
"""
from __future__ import annotations
import time
from core import feconf
from core.domain import feedback_services
from core.tests import test_utils
from typing import Final, TypedDict

class ExpectedThreadDict(TypedDict):
    """Type for the EXPECTED_THREAD_DICT dictionary."""
    status: str
    summary: None
    original_author_username: None
    subject: str

class FeedbackThreadSummariesLoadTests(test_utils.GenericTestBase):
    EXP_ID_1: Final = 'eid1'
    EXPECTED_THREAD_DICT: ExpectedThreadDict = {'status': u'open', 'summary': None, 'original_author_username': None, 'subject': u'a subject'}
    USER_EMAIL: Final = 'user@example.com'
    USER_USERNAME: Final = 'user'

    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        self.signup(self.USER_EMAIL, self.USER_USERNAME)
        self.signup(self.OWNER_EMAIL, self.OWNER_USERNAME)
        self.owner_id = self.get_user_id_from_email(self.OWNER_EMAIL)
        self.user_id = self.get_user_id_from_email(self.USER_EMAIL)
        self.save_new_valid_exploration(self.EXP_ID_1, self.owner_id, title='Bridges in England', category='Architecture', language_code='en')

    def test_get_thread_summaries_load_test(self) -> None:
        if False:
            return 10
        for _ in range(100):
            feedback_services.create_thread(feconf.ENTITY_TYPE_EXPLORATION, self.EXP_ID_1, self.user_id, self.EXPECTED_THREAD_DICT['subject'], 'not used here')
        threadlist = feedback_services.get_all_threads(feconf.ENTITY_TYPE_EXPLORATION, self.EXP_ID_1, False)
        thread_ids = []
        for thread in threadlist:
            thread_ids.append(thread.id)
            for _ in range(5):
                feedback_services.create_message(thread.id, self.user_id, None, None, 'editor message')
        start = time.time()
        feedback_services.get_exp_thread_summaries(self.user_id, thread_ids)
        elapsed_time = time.time() - start
        self.assertLessEqual(elapsed_time, 1.7)