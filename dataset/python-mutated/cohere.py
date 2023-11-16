from __future__ import annotations
from functools import cached_property
from typing import Any
import cohere
from airflow.hooks.base import BaseHook

class CohereHook(BaseHook):
    """
    Use Cohere Python SDK to interact with Cohere platform.

    .. seealso:: https://docs.cohere.com/docs

    :param conn_id: :ref:`Cohere connection id <howto/connection:cohere>`
    :param timeout: Request timeout in seconds.
    :param max_retries: Maximal number of retries for requests.
    """
    conn_name_attr = 'conn_id'
    default_conn_name = 'cohere_default'
    conn_type = 'cohere'
    hook_name = 'Cohere'

    def __init__(self, conn_id: str=default_conn_name, timeout: int | None=None, max_retries: int | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.conn_id = conn_id
        self.timeout = timeout
        self.max_retries = max_retries

    @cached_property
    def get_conn(self) -> cohere.Client:
        if False:
            while True:
                i = 10
        conn = self.get_connection(self.conn_id)
        return cohere.Client(api_key=conn.password, timeout=self.timeout, max_retries=self.max_retries, api_url=conn.host)

    def create_embeddings(self, texts: list[str], model: str='embed-multilingual-v2.0') -> list[list[float]]:
        if False:
            print('Hello World!')
        response = self.get_conn.embed(texts=texts, model=model)
        embeddings = response.embeddings
        return embeddings

    @staticmethod
    def get_ui_field_behaviour() -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return {'hidden_fields': ['schema', 'login', 'port', 'extra'], 'relabeling': {'password': 'API Key'}}

    def test_connection(self) -> tuple[bool, str]:
        if False:
            for i in range(10):
                print('nop')
        try:
            self.get_conn.generate('Test', max_tokens=10)
            return (True, 'Connection established')
        except Exception as e:
            return (False, str(e))