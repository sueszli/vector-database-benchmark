from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence
from airflow.models import BaseOperator
from airflow.providers.neo4j.hooks.neo4j import Neo4jHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class Neo4jOperator(BaseOperator):
    """
    Executes sql code in a specific Neo4j database.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:Neo4jOperator`

    :param sql: the sql code to be executed. Can receive a str representing a
        sql statement
    :param neo4j_conn_id: Reference to :ref:`Neo4j connection id <howto/connection:neo4j>`.
    """
    template_fields: Sequence[str] = ('sql',)

    def __init__(self, *, sql: str, neo4j_conn_id: str='neo4j_default', parameters: Iterable | Mapping[str, Any] | None=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.neo4j_conn_id = neo4j_conn_id
        self.sql = sql
        self.parameters = parameters

    def execute(self, context: Context) -> None:
        if False:
            while True:
                i = 10
        self.log.info('Executing: %s', self.sql)
        hook = Neo4jHook(conn_id=self.neo4j_conn_id)
        hook.run(self.sql)