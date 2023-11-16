import json
import time
from client import TestClient
from base import HubspotBaseTest

class TestHubspotTestClient(HubspotBaseTest):
    """
    Test the basic functionality of our Test Client. This is a tool for sanity checks, nothing more.

    To check an individual crud method, uncomment the corresponding test case below, and execute this file
    as if it is a normal tap-tester test via bin/run-test.
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.test_client = TestClient(self.get_properties()['start_date'])