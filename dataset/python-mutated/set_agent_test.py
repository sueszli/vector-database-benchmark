import os
from google.api_core.exceptions import InvalidArgument
import pytest
from set_agent import set_agent
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_set_agent():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidArgument):
        set_agent(PROJECT_ID, '')