from __future__ import annotations
from typing import TYPE_CHECKING
from botocore.waiter import Waiter, WaiterModel, create_waiter_with_client
if TYPE_CHECKING:
    import boto3

class BaseBotoWaiter:
    """
    Used to create custom Boto3 Waiters.

    For more details, see airflow/providers/amazon/aws/waiters/README.md
    """

    def __init__(self, client: boto3.client, model_config: dict, deferrable: bool=False) -> None:
        if False:
            print('Hello World!')
        self.model = WaiterModel(model_config)
        self.client = client
        self.deferrable = deferrable

    def _get_async_waiter_with_client(self, waiter_name: str):
        if False:
            for i in range(10):
                print('nop')
        from aiobotocore.waiter import create_waiter_with_client as create_async_waiter_with_client
        return create_async_waiter_with_client(waiter_name=waiter_name, waiter_model=self.model, client=self.client)

    def waiter(self, waiter_name: str) -> Waiter:
        if False:
            i = 10
            return i + 15
        if self.deferrable:
            return self._get_async_waiter_with_client(waiter_name=waiter_name)
        return create_waiter_with_client(waiter_name=waiter_name, waiter_model=self.model, client=self.client)