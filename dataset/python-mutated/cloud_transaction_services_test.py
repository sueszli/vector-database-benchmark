"""Unit tests for the cloud_transaction_services.py"""
from __future__ import annotations
from core.platform.transactions import cloud_transaction_services
from core.tests import test_utils

class CloudTransactionServicesTests(test_utils.GenericTestBase):
    """Unit tests for the cloud_transaction_services.py"""

    def test_run_in_transaction_wrapper(self) -> None:
        if False:
            print('Hello World!')
        calls_made = {'enter_context': False, 'exit_context': False}

        class MockTransaction:

            def __enter__(self) -> None:
                if False:
                    while True:
                        i = 10
                calls_made['enter_context'] = True

            def __exit__(self, *unused_args: str) -> None:
                if False:
                    while True:
                        i = 10
                calls_made['exit_context'] = True

        class MockClient:

            def transaction(self) -> MockTransaction:
                if False:
                    for i in range(10):
                        print('nop')
                return MockTransaction()
        swap_client = self.swap(cloud_transaction_services, 'CLIENT', MockClient())

        def add(x: int, y: int) -> int:
            if False:
                i = 10
                return i + 15
            return x + y
        with swap_client:
            wrapper_fn = cloud_transaction_services.run_in_transaction_wrapper(add)
            result = wrapper_fn(1, 2)
        self.assertEqual(result, 3)
        self.assertTrue(calls_made['enter_context'])
        self.assertTrue(calls_made['exit_context'])