import google.auth
from list_testcase_results import list_test_case
LOCATION = 'global'
(_, PROJECT_ID) = google.auth.default()
AGENT_ID = '143dee60-56fe-4191-a8d8-095f569f6cd8'
TEST_ID = '3c48d39e-71c0-4cb0-b974-3d5c596d347e'

def test_list_testcase_results():
    if False:
        for i in range(10):
            print('nop')
    result = list_test_case(PROJECT_ID, AGENT_ID, TEST_ID, LOCATION)
    assert 'Hello! How can I help you?' in str(result)