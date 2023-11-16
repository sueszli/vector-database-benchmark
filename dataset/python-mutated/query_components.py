import datetime
from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Dict, Hashable, List, Optional, Sequence, Tuple, Union, cast
from uuid import UUID
import pendulum
import sqlalchemy as sa
from cachetools import TTLCache
from jinja2 import Environment, PackageLoader, select_autoescape
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession
from prefect.server import schemas
from prefect.server.exceptions import FlowRunGraphTooLarge, ObjectNotFoundError
from prefect.server.schemas.graph import Edge, Graph, Node
from prefect.server.utilities.database import UUID as UUIDTypeDecorator
from prefect.server.utilities.database import Timestamp, json_has_any_key
if TYPE_CHECKING:
    from prefect.server.database.interface import PrefectDBInterface
ONE_HOUR = 60 * 60
jinja_env = Environment(loader=PackageLoader('prefect.server.database', package_path='sql'), autoescape=select_autoescape(), trim_blocks=True)

class BaseQueryComponents(ABC):
    """
    Abstract base class used to inject dialect-specific SQL operations into Prefect.
    """
    CONFIGURATION_CACHE = TTLCache(maxsize=100, ttl=ONE_HOUR)

    def _unique_key(self) -> Tuple[Hashable, ...]:
        if False:
            print('Hello World!')
        '\n        Returns a key used to determine whether to instantiate a new DB interface.\n        '
        return (self.__class__,)

    @abstractmethod
    def insert(self, obj):
        if False:
            i = 10
            return i + 15
        'dialect-specific insert statement'

    @abstractmethod
    def greatest(self, *values):
        if False:
            print('Hello World!')
        'dialect-specific SqlAlchemy binding'

    @abstractmethod
    def least(self, *values):
        if False:
            i = 10
            return i + 15
        'dialect-specific SqlAlchemy binding'

    @abstractproperty
    def uses_json_strings(self) -> bool:
        if False:
            print('Hello World!')
        'specifies whether the configured dialect returns JSON as strings'

    @abstractmethod
    def cast_to_json(self, json_obj):
        if False:
            for i in range(10):
                print('nop')
        'casts to JSON object if necessary'

    @abstractmethod
    def build_json_object(self, *args):
        if False:
            return 10
        'builds a JSON object from sequential key-value pairs'

    @abstractmethod
    def json_arr_agg(self, json_array):
        if False:
            return 10
        'aggregates a JSON array'

    @abstractmethod
    def make_timestamp_intervals(self, start_time: datetime.datetime, end_time: datetime.datetime, interval: datetime.timedelta):
        if False:
            while True:
                i = 10
        ...

    @abstractmethod
    def set_state_id_on_inserted_flow_runs_statement(self, fr_model, frs_model, inserted_flow_run_ids, insert_flow_run_states):
        if False:
            print('Hello World!')
        ...

    @abstractmethod
    async def get_flow_run_notifications_from_queue(self, session: AsyncSession, db: 'PrefectDBInterface', limit: int):
        """Database-specific implementation of reading notifications from the queue and deleting them"""

    async def queue_flow_run_notifications(self, session: sa.orm.session, flow_run: schemas.core.FlowRun, db: 'PrefectDBInterface'):
        """Database-specific implementation of queueing notifications for a flow run"""
        stmt = (await db.insert(db.FlowRunNotificationQueue)).from_select([db.FlowRunNotificationQueue.flow_run_notification_policy_id, db.FlowRunNotificationQueue.flow_run_state_id], sa.select(db.FlowRunNotificationPolicy.id, sa.cast(sa.literal(str(flow_run.state_id)), UUIDTypeDecorator)).select_from(db.FlowRunNotificationPolicy).where(sa.and_(db.FlowRunNotificationPolicy.is_active.is_(True), sa.or_(db.FlowRunNotificationPolicy.state_names == [], json_has_any_key(db.FlowRunNotificationPolicy.state_names, [flow_run.state_name])), sa.or_(db.FlowRunNotificationPolicy.tags == [], json_has_any_key(db.FlowRunNotificationPolicy.tags, flow_run.tags)))), include_defaults=False)
        await session.execute(stmt)

    def get_scheduled_flow_runs_from_work_queues(self, db: 'PrefectDBInterface', limit_per_queue: Optional[int]=None, work_queue_ids: Optional[List[UUID]]=None, scheduled_before: Optional[datetime.datetime]=None):
        if False:
            return 10
        '\n        Returns all scheduled runs in work queues, subject to provided parameters.\n\n        This query returns a `(db.FlowRun, db.WorkQueue.id)` pair; calling\n        `result.all()` will return both; calling `result.scalars().unique().all()`\n        will return only the flow run because it grabs the first result.\n        '
        concurrency_queues = sa.select(db.WorkQueue.id, self.greatest(0, db.WorkQueue.concurrency_limit - sa.func.count(db.FlowRun.id)).label('available_slots')).select_from(db.WorkQueue).join(db.FlowRun, sa.and_(self._flow_run_work_queue_join_clause(db.FlowRun, db.WorkQueue), db.FlowRun.state_type.in_(['RUNNING', 'PENDING', 'CANCELLING'])), isouter=True).where(db.WorkQueue.concurrency_limit.is_not(None)).group_by(db.WorkQueue.id).cte('concurrency_queues')
        (scheduled_flow_runs, join_criteria) = self._get_scheduled_flow_runs_join(db=db, work_queue_query=concurrency_queues, limit_per_queue=limit_per_queue, scheduled_before=scheduled_before)
        query = sa.select(sa.orm.aliased(db.FlowRun, scheduled_flow_runs), db.WorkQueue.id.label('wq_id')).select_from(db.WorkQueue).join(concurrency_queues, db.WorkQueue.id == concurrency_queues.c.id, isouter=True).join(scheduled_flow_runs, join_criteria).where(db.WorkQueue.is_paused.is_(False), db.WorkQueue.id.in_(work_queue_ids) if work_queue_ids else True).order_by(scheduled_flow_runs.c.next_scheduled_start_time, scheduled_flow_runs.c.id)
        return query

    def _get_scheduled_flow_runs_join(self, db: 'PrefectDBInterface', work_queue_query, limit_per_queue: Optional[int], scheduled_before: Optional[datetime.datetime]):
        if False:
            for i in range(10):
                print('nop')
        'Used by self.get_scheduled_flow_runs_from_work_queue, allowing just\n        this function to be changed on a per-dialect basis'
        scheduled_before_clause = db.FlowRun.next_scheduled_start_time <= scheduled_before if scheduled_before is not None else True
        scheduled_flow_runs = sa.select(db.FlowRun).where(self._flow_run_work_queue_join_clause(db.FlowRun, db.WorkQueue), db.FlowRun.state_type == 'SCHEDULED', scheduled_before_clause).with_for_update(skip_locked=True).order_by(db.FlowRun.next_scheduled_start_time).limit(sa.func.least(limit_per_queue, work_queue_query.c.available_slots)).lateral('scheduled_flow_runs')
        join_criteria = sa.literal(True)
        return (scheduled_flow_runs, join_criteria)

    def _flow_run_work_queue_join_clause(self, flow_run, work_queue):
        if False:
            while True:
                i = 10
        '\n        On clause for for joining flow runs to work queues\n\n        Used by self.get_scheduled_flow_runs_from_work_queue, allowing just\n        this function to be changed on a per-dialect basis\n        '
        return sa.and_(flow_run.work_queue_name == work_queue.name)

    @abstractproperty
    def _get_scheduled_flow_runs_from_work_pool_template_path(self):
        if False:
            return 10
        '\n        Template for the query to get scheduled flow runs from a work pool\n        '

    async def get_scheduled_flow_runs_from_work_pool(self, session, db: 'PrefectDBInterface', limit: Optional[int]=None, worker_limit: Optional[int]=None, queue_limit: Optional[int]=None, work_pool_ids: Optional[List[UUID]]=None, work_queue_ids: Optional[List[UUID]]=None, scheduled_before: Optional[datetime.datetime]=None, scheduled_after: Optional[datetime.datetime]=None, respect_queue_priorities: bool=False) -> List[schemas.responses.WorkerFlowRunResponse]:
        template = jinja_env.get_template(self._get_scheduled_flow_runs_from_work_pool_template_path)
        raw_query = sa.text(template.render(work_pool_ids=work_pool_ids, work_queue_ids=work_queue_ids, respect_queue_priorities=respect_queue_priorities, scheduled_before=scheduled_before, scheduled_after=scheduled_after))
        bindparams = []
        if scheduled_before:
            bindparams.append(sa.bindparam('scheduled_before', scheduled_before, type_=Timestamp))
        if scheduled_after:
            bindparams.append(sa.bindparam('scheduled_after', scheduled_after, type_=Timestamp))
        if work_pool_ids:
            assert all((isinstance(i, UUID) for i in work_pool_ids))
            bindparams.append(sa.bindparam('work_pool_ids', work_pool_ids, expanding=True, type_=UUIDTypeDecorator))
        if work_queue_ids:
            assert all((isinstance(i, UUID) for i in work_queue_ids))
            bindparams.append(sa.bindparam('work_queue_ids', work_queue_ids, expanding=True, type_=UUIDTypeDecorator))
        query = raw_query.bindparams(*bindparams, limit=1000 if limit is None else limit, worker_limit=1000 if worker_limit is None else worker_limit, queue_limit=1000 if queue_limit is None else queue_limit)
        orm_query = sa.select(sa.column('run_work_pool_id'), sa.column('run_work_queue_id'), db.FlowRun).from_statement(query).options(sa.orm.noload(db.FlowRun.state))
        result = await session.execute(orm_query)
        return [schemas.responses.WorkerFlowRunResponse(work_pool_id=r.run_work_pool_id, work_queue_id=r.run_work_queue_id, flow_run=schemas.core.FlowRun.from_orm(r.FlowRun)) for r in result]

    async def read_block_documents(self, session: sa.orm.Session, db: 'PrefectDBInterface', block_document_filter: Optional[schemas.filters.BlockDocumentFilter]=None, block_type_filter: Optional[schemas.filters.BlockTypeFilter]=None, block_schema_filter: Optional[schemas.filters.BlockSchemaFilter]=None, include_secrets: bool=False, offset: Optional[int]=None, limit: Optional[int]=None):
        if block_document_filter is None:
            block_document_filter = schemas.filters.BlockDocumentFilter(is_anonymous=schemas.filters.BlockDocumentFilterIsAnonymous(eq_=False))
        filtered_block_documents_query = sa.select(db.BlockDocument.id).where(block_document_filter.as_sql_filter(db))
        if block_type_filter is not None:
            block_type_exists_clause = sa.select(db.BlockType).where(db.BlockType.id == db.BlockDocument.block_type_id, block_type_filter.as_sql_filter(db))
            filtered_block_documents_query = filtered_block_documents_query.where(block_type_exists_clause.exists())
        if block_schema_filter is not None:
            block_schema_exists_clause = sa.select(db.BlockSchema).where(db.BlockSchema.id == db.BlockDocument.block_schema_id, block_schema_filter.as_sql_filter(db))
            filtered_block_documents_query = filtered_block_documents_query.where(block_schema_exists_clause.exists())
        if offset is not None:
            filtered_block_documents_query = filtered_block_documents_query.offset(offset)
        if limit is not None:
            filtered_block_documents_query = filtered_block_documents_query.limit(limit)
        filtered_block_documents_query = filtered_block_documents_query.cte('filtered_block_documents')
        block_document_references_query = sa.select(db.BlockDocumentReference).filter(db.BlockDocumentReference.parent_block_document_id.in_(sa.select(filtered_block_documents_query.c.id))).cte('block_document_references', recursive=True)
        block_document_references_join = sa.select(db.BlockDocumentReference).join(block_document_references_query, db.BlockDocumentReference.parent_block_document_id == block_document_references_query.c.reference_block_document_id)
        recursive_block_document_references_cte = block_document_references_query.union_all(block_document_references_join)
        all_block_documents_query = sa.union_all(sa.select(db.BlockDocument, sa.null().label('reference_name'), sa.null().label('reference_parent_block_document_id')).select_from(db.BlockDocument).where(db.BlockDocument.id.in_(sa.select(filtered_block_documents_query.c.id))), sa.select(db.BlockDocument, recursive_block_document_references_cte.c.name, recursive_block_document_references_cte.c.parent_block_document_id).select_from(db.BlockDocument).join(recursive_block_document_references_cte, db.BlockDocument.id == recursive_block_document_references_cte.c.reference_block_document_id)).cte('all_block_documents_query')
        return sa.select(sa.orm.aliased(db.BlockDocument, all_block_documents_query), all_block_documents_query.c.reference_name, all_block_documents_query.c.reference_parent_block_document_id).select_from(all_block_documents_query).order_by(all_block_documents_query.c.name)

    async def read_configuration_value(self, db: 'PrefectDBInterface', session: sa.orm.Session, key: str) -> Optional[Dict]:
        """
        Read a configuration value by key.

        Configuration values should not be changed at run time, so retrieved
        values are cached in memory.

        The main use of configurations is encrypting blocks, this speeds up nested
        block document queries.
        """
        try:
            return self.CONFIGURATION_CACHE[key]
        except KeyError:
            query = sa.select(db.Configuration).where(db.Configuration.key == key)
            result = await session.execute(query)
            configuration = result.scalar()
            if configuration is not None:
                self.CONFIGURATION_CACHE[key] = configuration.value
                return configuration.value
            return configuration

    def clear_configuration_value_cache_for_key(self, key: str):
        if False:
            return 10
        'Removes a configuration key from the cache.'
        self.CONFIGURATION_CACHE.pop(key, None)

    @abstractmethod
    async def flow_run_graph_v2(self, db: 'PrefectDBInterface', session: AsyncSession, flow_run_id: UUID, since: datetime.datetime, max_nodes: int) -> Graph:
        """Returns the query that selects all of the nodes and edges for a flow run graph (version 2)."""
        ...

class AsyncPostgresQueryComponents(BaseQueryComponents):

    def insert(self, obj):
        if False:
            i = 10
            return i + 15
        return postgresql.insert(obj)

    def greatest(self, *values):
        if False:
            return 10
        return sa.func.greatest(*values)

    def least(self, *values):
        if False:
            i = 10
            return i + 15
        return sa.func.least(*values)

    @property
    def uses_json_strings(self):
        if False:
            while True:
                i = 10
        return False

    def cast_to_json(self, json_obj):
        if False:
            for i in range(10):
                print('nop')
        return json_obj

    def build_json_object(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return sa.func.jsonb_build_object(*args)

    def json_arr_agg(self, json_array):
        if False:
            print('Hello World!')
        return sa.func.jsonb_agg(json_array)

    def make_timestamp_intervals(self, start_time: datetime.datetime, end_time: datetime.datetime, interval: datetime.timedelta):
        if False:
            print('Hello World!')
        start_time = pendulum.instance(start_time)
        end_time = pendulum.instance(end_time)
        assert isinstance(interval, datetime.timedelta)
        return sa.select(sa.literal_column('dt').label('interval_start'), (sa.literal_column('dt') + interval).label('interval_end')).select_from(sa.func.generate_series(start_time, end_time, interval).alias('dt')).where(sa.literal_column('dt') < end_time).limit(500)

    def set_state_id_on_inserted_flow_runs_statement(self, fr_model, frs_model, inserted_flow_run_ids, insert_flow_run_states):
        if False:
            return 10
        'Given a list of flow run ids and associated states, set the state_id\n        to the appropriate state for all flow runs'
        stmt = sa.update(fr_model).where(fr_model.id.in_(inserted_flow_run_ids), frs_model.flow_run_id == fr_model.id, frs_model.id.in_([r['id'] for r in insert_flow_run_states])).values(state_id=frs_model.id).execution_options(synchronize_session=False)
        return stmt

    async def get_flow_run_notifications_from_queue(self, session: AsyncSession, db: 'PrefectDBInterface', limit: int) -> List:
        queued_notifications_ids = sa.select(db.FlowRunNotificationQueue.id).select_from(db.FlowRunNotificationQueue).order_by(db.FlowRunNotificationQueue.updated).limit(limit).with_for_update(skip_locked=True).cte('queued_notifications_ids')
        queued_notifications = sa.delete(db.FlowRunNotificationQueue).returning(db.FlowRunNotificationQueue.id, db.FlowRunNotificationQueue.flow_run_notification_policy_id, db.FlowRunNotificationQueue.flow_run_state_id).where(db.FlowRunNotificationQueue.id.in_(sa.select(queued_notifications_ids))).cte('queued_notifications')
        notification_details_stmt = sa.select(queued_notifications.c.id.label('queue_id'), db.FlowRunNotificationPolicy.id.label('flow_run_notification_policy_id'), db.FlowRunNotificationPolicy.message_template.label('flow_run_notification_policy_message_template'), db.FlowRunNotificationPolicy.block_document_id, db.Flow.id.label('flow_id'), db.Flow.name.label('flow_name'), db.FlowRun.id.label('flow_run_id'), db.FlowRun.name.label('flow_run_name'), db.FlowRun.parameters.label('flow_run_parameters'), db.FlowRunState.type.label('flow_run_state_type'), db.FlowRunState.name.label('flow_run_state_name'), db.FlowRunState.timestamp.label('flow_run_state_timestamp'), db.FlowRunState.message.label('flow_run_state_message')).select_from(queued_notifications).join(db.FlowRunNotificationPolicy, queued_notifications.c.flow_run_notification_policy_id == db.FlowRunNotificationPolicy.id).join(db.FlowRunState, queued_notifications.c.flow_run_state_id == db.FlowRunState.id).join(db.FlowRun, db.FlowRunState.flow_run_id == db.FlowRun.id).join(db.Flow, db.FlowRun.flow_id == db.Flow.id)
        result = await session.execute(notification_details_stmt)
        return result.fetchall()

    @property
    def _get_scheduled_flow_runs_from_work_pool_template_path(self):
        if False:
            print('Hello World!')
        '\n        Template for the query to get scheduled flow runs from a work pool\n        '
        return 'postgres/get-runs-from-worker-queues.sql.jinja'

    async def flow_run_graph_v2(self, db: 'PrefectDBInterface', session: AsyncSession, flow_run_id: UUID, since: datetime.datetime, max_nodes: int) -> Graph:
        """Returns the query that selects all of the nodes and edges for a flow run
        graph (version 2)."""
        result = await session.execute(sa.select(sa.func.coalesce(db.FlowRun.start_time, db.FlowRun.expected_start_time), db.FlowRun.end_time).where(db.FlowRun.id == flow_run_id))
        try:
            (start_time, end_time) = result.one()
        except NoResultFound:
            raise ObjectNotFoundError(f'Flow run {flow_run_id} not found')
        query = sa.text("\n            WITH\n            edges AS (\n                SELECT  CASE\n                            WHEN subflow.id IS NOT NULL THEN 'flow-run'\n                            ELSE 'task-run'\n                        END as kind,\n                        COALESCE(subflow.id, task_run.id) as id,\n                        COALESCE(flow.name || ' / ' || subflow.name, task_run.name) as label,\n                        COALESCE(subflow.state_type, task_run.state_type) as state_type,\n                        COALESCE(\n                            subflow.start_time,\n                            subflow.expected_start_time,\n                            task_run.start_time,\n                            task_run.expected_start_time\n                        ) as start_time,\n                        COALESCE(\n                            subflow.end_time,\n                            task_run.end_time,\n                            CASE\n                                WHEN task_run.state_type = 'COMPLETED'\n                                    THEN task_run.expected_start_time\n                                ELSE NULL\n                            END\n                        ) as end_time,\n                        (argument->>'id')::uuid as parent\n                FROM    task_run\n                        LEFT JOIN jsonb_each(task_run.task_inputs) as input ON true\n                        LEFT JOIN jsonb_array_elements(input.value) as argument ON true\n                        LEFT JOIN flow_run as subflow\n                                ON subflow.parent_task_run_id = task_run.id\n                        LEFT JOIN flow\n                                ON flow.id = subflow.flow_id\n                WHERE   task_run.flow_run_id = :flow_run_id AND\n                        task_run.state_type <> 'PENDING' AND\n                        COALESCE(\n                            subflow.start_time,\n                            subflow.expected_start_time,\n                            task_run.start_time,\n                            task_run.expected_start_time\n                        ) IS NOT NULL\n\n                -- the order here is important to speed up building the two sets of\n                -- edges in the with_parents and with_children CTEs below\n                ORDER BY COALESCE(subflow.id, task_run.id)\n            ),\n            with_parents AS (\n                SELECT  children.id,\n                        array_agg(parents.id order by parents.start_time) as parent_ids\n                FROM    edges as children\n                        INNER JOIN edges as parents\n                                ON parents.id = children.parent\n                GROUP BY children.id\n            ),\n            with_children AS (\n                SELECT  parents.id,\n                        array_agg(children.id order by children.start_time) as child_ids\n                FROM    edges as parents\n                        INNER JOIN edges as children\n                                ON children.parent = parents.id\n                GROUP BY parents.id\n            ),\n            nodes AS (\n                SELECT  DISTINCT ON (edges.id)\n                        edges.kind,\n                        edges.id,\n                        edges.label,\n                        edges.state_type,\n                        edges.start_time,\n                        edges.end_time,\n                        with_parents.parent_ids,\n                        with_children.child_ids\n                FROM    edges\n                        LEFT JOIN with_parents\n                                ON with_parents.id = edges.id\n                        LEFT JOIN with_children\n                                ON with_children.id = edges.id\n            )\n            SELECT  kind,\n                    id,\n                    label,\n                    state_type,\n                    start_time,\n                    end_time,\n                    parent_ids,\n                    child_ids\n            FROM    nodes\n            WHERE   end_time IS NULL OR end_time >= :since\n            ORDER BY start_time, end_time\n            LIMIT :max_nodes\n            ;\n        ")
        query = query.bindparams(sa.bindparam('flow_run_id', value=flow_run_id), sa.bindparam('since', value=since), sa.bindparam('max_nodes', value=max_nodes + 1))
        results = await session.execute(query)
        nodes: List[Tuple[UUID, Node]] = []
        root_node_ids: List[UUID] = []
        for row in results:
            if not row.parent_ids:
                root_node_ids.append(row.id)
            nodes.append((row.id, Node(kind=row.kind, id=row.id, label=row.label, state_type=row.state_type, start_time=row.start_time, end_time=row.end_time, parents=[Edge(id=id) for id in row.parent_ids or []], children=[Edge(id=id) for id in row.child_ids or []])))
            if len(nodes) > max_nodes:
                raise FlowRunGraphTooLarge(f'The graph of flow run {flow_run_id} has more than {max_nodes} nodes.')
        return Graph(start_time=start_time, end_time=end_time, root_node_ids=root_node_ids, nodes=nodes)

class AioSqliteQueryComponents(BaseQueryComponents):

    def insert(self, obj):
        if False:
            while True:
                i = 10
        return sqlite.insert(obj)

    def greatest(self, *values):
        if False:
            for i in range(10):
                print('nop')
        return sa.func.max(*values)

    def least(self, *values):
        if False:
            while True:
                i = 10
        return sa.func.min(*values)

    @property
    def uses_json_strings(self):
        if False:
            while True:
                i = 10
        return True

    def cast_to_json(self, json_obj):
        if False:
            while True:
                i = 10
        return sa.func.json(json_obj)

    def build_json_object(self, *args):
        if False:
            return 10
        return sa.func.json_object(*args)

    def json_arr_agg(self, json_array):
        if False:
            print('Hello World!')
        return sa.func.json_group_array(json_array)

    def make_timestamp_intervals(self, start_time: datetime.datetime, end_time: datetime.datetime, interval: datetime.timedelta):
        if False:
            return 10
        from prefect.server.utilities.database import Timestamp
        start_time = pendulum.instance(start_time)
        end_time = pendulum.instance(end_time)
        assert isinstance(interval, datetime.timedelta)
        return sa.text("\n                -- recursive CTE to mimic the behavior of `generate_series`,\n                -- which is only available as a compiled extension\n                WITH RECURSIVE intervals(interval_start, interval_end, counter) AS (\n                    VALUES(\n                        strftime('%Y-%m-%d %H:%M:%f000', :start_time),\n                        strftime('%Y-%m-%d %H:%M:%f000', :start_time, :interval),\n                        1\n                        )\n\n                    UNION ALL\n\n                    SELECT interval_end, strftime('%Y-%m-%d %H:%M:%f000', interval_end, :interval), counter + 1\n                    FROM intervals\n                    -- subtract interval because recursive where clauses are effectively evaluated on a t-1 lag\n                    WHERE\n                        interval_start < strftime('%Y-%m-%d %H:%M:%f000', :end_time, :negative_interval)\n                        -- don't compute more than 500 intervals\n                        AND counter < 500\n                )\n                SELECT * FROM intervals\n                ").bindparams(start_time=str(start_time), end_time=str(end_time), interval=f'+{interval.total_seconds()} seconds', negative_interval=f'-{interval.total_seconds()} seconds').columns(interval_start=Timestamp(), interval_end=Timestamp())

    def set_state_id_on_inserted_flow_runs_statement(self, fr_model, frs_model, inserted_flow_run_ids, insert_flow_run_states):
        if False:
            print('Hello World!')
        'Given a list of flow run ids and associated states, set the state_id\n        to the appropriate state for all flow runs'
        subquery = sa.select(frs_model.id).where(frs_model.flow_run_id == fr_model.id, frs_model.id.in_([r['id'] for r in insert_flow_run_states])).limit(1).scalar_subquery()
        stmt = sa.update(fr_model).where(fr_model.id.in_(inserted_flow_run_ids)).values(state_id=subquery).execution_options(synchronize_session=False)
        return stmt

    async def get_flow_run_notifications_from_queue(self, session: AsyncSession, db: 'PrefectDBInterface', limit: int) -> List:
        """
        Sqlalchemy has no support for DELETE RETURNING in sqlite (as of May 2022)
        so instead we issue two queries; one to get queued notifications and a second to delete
        them. This *could* introduce race conditions if multiple queue workers are
        running.
        """
        notification_details_stmt = sa.select(db.FlowRunNotificationQueue.id.label('queue_id'), db.FlowRunNotificationPolicy.id.label('flow_run_notification_policy_id'), db.FlowRunNotificationPolicy.message_template.label('flow_run_notification_policy_message_template'), db.FlowRunNotificationPolicy.block_document_id, db.Flow.id.label('flow_id'), db.Flow.name.label('flow_name'), db.FlowRun.id.label('flow_run_id'), db.FlowRun.name.label('flow_run_name'), db.FlowRun.parameters.label('flow_run_parameters'), db.FlowRunState.type.label('flow_run_state_type'), db.FlowRunState.name.label('flow_run_state_name'), db.FlowRunState.timestamp.label('flow_run_state_timestamp'), db.FlowRunState.message.label('flow_run_state_message')).select_from(db.FlowRunNotificationQueue).join(db.FlowRunNotificationPolicy, db.FlowRunNotificationQueue.flow_run_notification_policy_id == db.FlowRunNotificationPolicy.id).join(db.FlowRunState, db.FlowRunNotificationQueue.flow_run_state_id == db.FlowRunState.id).join(db.FlowRun, db.FlowRunState.flow_run_id == db.FlowRun.id).join(db.Flow, db.FlowRun.flow_id == db.Flow.id).order_by(db.FlowRunNotificationQueue.updated).limit(limit)
        result = await session.execute(notification_details_stmt)
        notifications = result.fetchall()
        delete_stmt = sa.delete(db.FlowRunNotificationQueue).where(db.FlowRunNotificationQueue.id.in_([n.queue_id for n in notifications])).execution_options(synchronize_session='fetch')
        await session.execute(delete_stmt)
        return notifications

    async def _handle_filtered_block_document_ids(self, session, filtered_block_documents_query):
        """
        On SQLite, including the filtered block document parameters confuses the
        compiler and it passes positional parameters in the wrong order (it is
        unclear why; SQLalchemy manual compilation works great. Switching to
        `named` paramstyle also works but fails elsewhere in the codebase). To
        resolve this, we materialize the filtered id query into a literal set of
        IDs rather than leaving it as a SQL select.
        """
        result = await session.execute(filtered_block_documents_query)
        return result.scalars().all()

    def _get_scheduled_flow_runs_join(self, db: 'PrefectDBInterface', work_queue_query, limit_per_queue: Optional[int], scheduled_before: Optional[datetime.datetime]):
        if False:
            while True:
                i = 10
        scheduled_before_clause = db.FlowRun.next_scheduled_start_time <= scheduled_before if scheduled_before is not None else True
        scheduled_flow_runs = sa.select(sa.func.row_number().over(partition_by=[db.FlowRun.work_queue_name], order_by=db.FlowRun.next_scheduled_start_time).label('rank'), db.FlowRun).where(db.FlowRun.state_type == 'SCHEDULED', scheduled_before_clause).subquery('scheduled_flow_runs')
        limit = 999999 if limit_per_queue is None else limit_per_queue
        join_criteria = sa.and_(self._flow_run_work_queue_join_clause(scheduled_flow_runs.c, db.WorkQueue), scheduled_flow_runs.c.rank <= sa.func.min(sa.func.coalesce(work_queue_query.c.available_slots, limit), limit))
        return (scheduled_flow_runs, join_criteria)

    @property
    def _get_scheduled_flow_runs_from_work_pool_template_path(self):
        if False:
            return 10
        '\n        Template for the query to get scheduled flow runs from a work pool\n        '
        return 'sqlite/get-runs-from-worker-queues.sql.jinja'

    async def flow_run_graph_v2(self, db: 'PrefectDBInterface', session: AsyncSession, flow_run_id: UUID, since: datetime.datetime, max_nodes: int) -> Graph:
        """Returns the query that selects all of the nodes and edges for a flow run
        graph (version 2)."""
        result = await session.execute(sa.select(sa.func.coalesce(db.FlowRun.start_time, db.FlowRun.expected_start_time), db.FlowRun.end_time).where(db.FlowRun.id == flow_run_id))
        try:
            (start_time, end_time) = result.one()
        except NoResultFound:
            raise ObjectNotFoundError(f'Flow run {flow_run_id} not found')
        query = sa.text("\n            WITH\n            edges AS (\n                SELECT  CASE\n                            WHEN subflow.id IS NOT NULL THEN 'flow-run'\n                            ELSE 'task-run'\n                        END as kind,\n                        COALESCE(subflow.id, task_run.id) as id,\n                        COALESCE(flow.name || ' / ' || subflow.name, task_run.name) as label,\n                        COALESCE(subflow.state_type, task_run.state_type) as state_type,\n                        COALESCE(\n                            subflow.start_time,\n                            subflow.expected_start_time,\n                            task_run.start_time,\n                            task_run.expected_start_time\n                        ) as start_time,\n                        COALESCE(\n                            subflow.end_time,\n                            task_run.end_time,\n                            CASE\n                                WHEN task_run.state_type = 'COMPLETED'\n                                    THEN task_run.expected_start_time\n                                ELSE NULL\n                            END\n                        ) as end_time,\n                        json_extract(argument.value, '$.id') as parent\n                FROM    task_run\n                        LEFT JOIN json_each(task_run.task_inputs) as input ON true\n                        LEFT JOIN json_each(input.value) as argument ON true\n                        LEFT JOIN flow_run as subflow\n                                ON subflow.parent_task_run_id = task_run.id\n                        LEFT JOIN flow\n                                ON flow.id = subflow.flow_id\n                WHERE   task_run.flow_run_id = :flow_run_id AND\n                        task_run.state_type <> 'PENDING' AND\n                        COALESCE(\n                            subflow.start_time,\n                            subflow.expected_start_time,\n                            task_run.start_time,\n                            task_run.expected_start_time\n                        ) IS NOT NULL\n\n                -- the order here is important to speed up building the two sets of\n                -- edges in the with_parents and with_children CTEs below\n                ORDER BY COALESCE(subflow.id, task_run.id)\n            ),\n            with_parents AS (\n                SELECT  children.id,\n                        group_concat(parents.id) as parent_ids\n                FROM    edges as children\n                        INNER JOIN edges as parents\n                                ON parents.id = children.parent\n                GROUP BY children.id\n            ),\n            with_children AS (\n                SELECT  parents.id,\n                        group_concat(children.id) as child_ids\n                FROM    edges as parents\n                        INNER JOIN edges as children\n                                ON children.parent = parents.id\n                GROUP BY parents.id\n            ),\n            nodes AS (\n                SELECT  DISTINCT\n                        edges.id,\n                        edges.kind,\n                        edges.id,\n                        edges.label,\n                        edges.state_type,\n                        edges.start_time,\n                        edges.end_time,\n                        with_parents.parent_ids,\n                        with_children.child_ids\n                FROM    edges\n                        LEFT JOIN with_parents\n                                ON with_parents.id = edges.id\n                        LEFT JOIN with_children\n                                ON with_children.id = edges.id\n            )\n            SELECT  kind,\n                    id,\n                    label,\n                    state_type,\n                    start_time,\n                    end_time,\n                    parent_ids,\n                    child_ids\n            FROM    nodes\n            WHERE   end_time IS NULL OR end_time >= :since\n            ORDER BY start_time, end_time\n            LIMIT :max_nodes\n            ;\n        ")
        since = datetime.datetime(since.year, since.month, since.day, since.hour, since.minute, since.second, since.microsecond, tzinfo=since.tzinfo)
        query = query.bindparams(sa.bindparam('flow_run_id', value=str(flow_run_id)), sa.bindparam('since', value=since), sa.bindparam('max_nodes', value=max_nodes + 1))
        results = await session.execute(query)
        nodes: List[Tuple[UUID, Node]] = []
        root_node_ids: List[UUID] = []
        for row in results:
            if not row.parent_ids:
                root_node_ids.append(row.id)

            def edges(value: Union[str, Sequence[UUID], Sequence[str], None]) -> List[UUID]:
                if False:
                    return 10
                if not value:
                    return []
                if isinstance(value, str):
                    return [Edge(id=id) for id in value.split(',')]
                return [Edge(id=id) for id in value]

            def time(value: Union[str, datetime.datetime, None]) -> Optional[pendulum.DateTime]:
                if False:
                    return 10
                if not value:
                    return None
                if isinstance(value, str):
                    return cast(pendulum.DateTime, pendulum.parse(value))
                return pendulum.instance(value)
            nodes.append((row.id, Node(kind=row.kind, id=row.id, label=row.label, state_type=row.state_type, start_time=time(row.start_time), end_time=time(row.end_time), parents=edges(row.parent_ids), children=edges(row.child_ids))))
            if len(nodes) > max_nodes:
                raise FlowRunGraphTooLarge(f'The graph of flow run {flow_run_id} has more than {max_nodes} nodes.')
        return Graph(start_time=start_time, end_time=end_time, root_node_ids=root_node_ids, nodes=nodes)