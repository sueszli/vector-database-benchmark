import datetime
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterator, List, Optional, Set
from pony import orm
from pony.orm import raw_sql
from pony.orm.core import Entity, Query, select
from pony.utils import between
from tribler.core.components.knowledge.community.knowledge_payload import StatementOperation
from tribler.core.utilities.pony_utils import get_or_create, iterable
CLOCK_START_VALUE = 0
PUBLIC_KEY_FOR_AUTO_GENERATED_OPERATIONS = b'auto_generated'
SHOW_THRESHOLD = 1
HIDE_THRESHOLD = -2

class Operation(IntEnum):
    """ Available types of statement operations."""
    ADD = 1
    REMOVE = 2

class ResourceType(IntEnum):
    """ Description of available resources within the Knowledge Graph.
    These types are also using as a predicate for the statements.

    Based on https://en.wikipedia.org/wiki/Dublin_Core
    """
    CONTRIBUTOR = 1
    COVERAGE = 2
    CREATOR = 3
    DATE = 4
    DESCRIPTION = 5
    FORMAT = 6
    IDENTIFIER = 7
    LANGUAGE = 8
    PUBLISHER = 9
    RELATION = 10
    RIGHTS = 11
    SOURCE = 12
    SUBJECT = 13
    TITLE = 14
    TYPE = 15
    TAG = 101
    TORRENT = 102
    CONTENT_ITEM = 103

@dataclass
class SimpleStatement:
    subject_type: ResourceType
    object: str
    predicate: ResourceType
    subject: str

class KnowledgeDataAccessLayer:

    def __init__(self, instance: orm.Database):
        if False:
            i = 10
            return i + 15
        self.logger = logging.getLogger(self.__class__.__name__)
        self.instance = instance
        (self.Peer, self.Statement, self.Resource, self.StatementOp, self._get_statements) = self.define_binding(self.instance)

    @staticmethod
    def define_binding(db):
        if False:
            i = 10
            return i + 15

        class Peer(db.Entity):
            id = orm.PrimaryKey(int, auto=True)
            public_key = orm.Required(bytes, unique=True)
            added_at = orm.Optional(datetime.datetime, default=datetime.datetime.utcnow)
            operations = orm.Set(lambda : StatementOp)

        class Statement(db.Entity):
            id = orm.PrimaryKey(int, auto=True)
            subject = orm.Required(lambda : Resource)
            object = orm.Required(lambda : Resource, index=True)
            operations = orm.Set(lambda : StatementOp)
            added_count = orm.Required(int, default=0)
            removed_count = orm.Required(int, default=0)
            local_operation = orm.Optional(int)
            orm.composite_key(subject, object)

            @property
            def score(self):
                if False:
                    i = 10
                    return i + 15
                return self.added_count - self.removed_count

            def update_counter(self, operation: Operation, increment: int=1, is_local_peer: bool=False):
                if False:
                    return 10
                " Update Statement's counter\n                Args:\n                    operation: Resource operation\n                    increment:\n                    is_local_peer: The flag indicates whether do we perform operations from a local user or from\n                        a remote user. In case of the local user, his operations will be considered as\n                        authoritative for his (only) local Tribler instance.\n\n                Returns:\n                "
                if is_local_peer:
                    self.local_operation = operation
                if operation == Operation.ADD:
                    self.added_count += increment
                if operation == Operation.REMOVE:
                    self.removed_count += increment

        class Resource(db.Entity):
            id = orm.PrimaryKey(int, auto=True)
            name = orm.Required(str)
            type = orm.Required(int)
            subject_statements = orm.Set(lambda : Statement, reverse='subject')
            object_statements = orm.Set(lambda : Statement, reverse='object')
            torrent_healths = orm.Set(lambda : db.TorrentHealth, reverse='torrent')
            trackers = orm.Set(lambda : db.Tracker, reverse='torrents')
            orm.composite_key(name, type)

        class StatementOp(db.Entity):
            id = orm.PrimaryKey(int, auto=True)
            statement = orm.Required(lambda : Statement)
            peer = orm.Required(lambda : Peer)
            operation = orm.Required(int)
            clock = orm.Required(int)
            signature = orm.Required(bytes)
            updated_at = orm.Required(datetime.datetime, default=datetime.datetime.utcnow)
            auto_generated = orm.Required(bool, default=False)
            orm.composite_key(statement, peer)

        def _get_resources(resource_type: Optional[ResourceType], name: Optional[str], case_sensitive: bool) -> Query:
            if False:
                i = 10
                return i + 15
            ' Get resources\n\n            Args:\n                resource_type: type of resources\n                name: name of resources\n                case_sensitive: if True, then Resources are selected in a case-sensitive manner.\n                                if False, then Resources are selected in a case-insensitive manner.\n\n            Returns: a Query object for requested resources\n            '
            results = Resource.select()
            if name:
                results = results.filter((lambda r: r.name == name) if case_sensitive else lambda r: r.name.lower() == name.lower())
            if resource_type:
                results = results.filter(lambda r: r.type == resource_type.value)
            return results

        def _get_statements(source_type: Optional[ResourceType], source_name: Optional[str], statements_getter: Callable[[Entity], Entity], target_condition: Callable[[], bool], condition: Callable[[], bool], case_sensitive: bool) -> Iterator[Statement]:
            if False:
                while True:
                    i = 10
            ' Get entities that satisfies the given condition.\n            '
            for resource in _get_resources(source_type, source_name, case_sensitive):
                results = orm.select((_ for _ in statements_getter(resource).select(condition).filter(target_condition).order_by(lambda s: orm.desc(s.score))))
                yield from list(results)
        return (Peer, Statement, Resource, StatementOp, _get_statements)

    def add_operation(self, operation: StatementOperation, signature: bytes, is_local_peer: bool=False, is_auto_generated: bool=False, counter_increment: int=1) -> bool:
        if False:
            print('Hello World!')
        ' Add the operation that will be applied to a statement.\n        Args:\n            operation: the class describes the adding operation\n            signature: the signature of the operation\n            is_local_peer: local operations processes differently than remote operations. They affects\n                `Statement.local_operation` field which is used in `self.get_tags()` function.\n            is_auto_generated: the indicator of whether this resource was generated automatically or not\n            counter_increment: the counter or "numbers" of adding operations\n\n        Returns: True if the operation has been added/updated, False otherwise.\n        '
        self.logger.debug(f'Add operation. {operation.subject} "{operation.predicate}" {operation.object}')
        peer = get_or_create(self.Peer, public_key=operation.creator_public_key)
        subject = get_or_create(self.Resource, name=operation.subject, type=operation.subject_type)
        obj = get_or_create(self.Resource, name=operation.object, type=operation.predicate)
        statement = get_or_create(self.Statement, subject=subject, object=obj)
        op = self.StatementOp.get_for_update(statement=statement, peer=peer)
        if not op:
            self.StatementOp(statement=statement, peer=peer, operation=operation.operation, clock=operation.clock, signature=signature, auto_generated=is_auto_generated)
            statement.update_counter(operation.operation, increment=counter_increment, is_local_peer=is_local_peer)
            return True
        if operation.clock <= op.clock:
            return False
        statement.update_counter(op.operation, increment=-counter_increment, is_local_peer=is_local_peer)
        statement.update_counter(operation.operation, increment=counter_increment, is_local_peer=is_local_peer)
        op.set(operation=operation.operation, clock=operation.clock, signature=signature, updated_at=datetime.datetime.utcnow(), auto_generated=is_auto_generated)
        return True

    def add_auto_generated_operation(self, subject_type: ResourceType, subject: str, predicate: ResourceType, obj: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ' Add an autogenerated operation.\n\n        The difference between "normal" and "autogenerated" operation is that the  autogenerated operation will be added\n        with the flag `is_auto_generated=True` and with the `PUBLIC_KEY_FOR_AUTO_GENERATED_TAGS` public key.\n\n        Args:\n            subject_type: a type of adding subject. See: ResourceType enum.\n            subject: a string that represents a subject of adding operation.\n            predicate: the enum that represents a predicate of adding operation.\n            obj: a string that represents an object of adding operation.\n        '
        operation = StatementOperation(subject_type=subject_type, subject=subject, predicate=predicate, object=obj, operation=Operation.ADD, clock=CLOCK_START_VALUE, creator_public_key=PUBLIC_KEY_FOR_AUTO_GENERATED_OPERATIONS)
        return self.add_operation(operation, signature=b'', is_local_peer=False, is_auto_generated=True, counter_increment=SHOW_THRESHOLD)

    @staticmethod
    def _show_condition(s):
        if False:
            return 10
        'This function determines show condition for the statement'
        return s.local_operation == Operation.ADD.value or (not s.local_operation and s.score >= SHOW_THRESHOLD)

    def get_objects(self, subject_type: Optional[ResourceType]=None, subject: Optional[str]='', predicate: Optional[ResourceType]=None, case_sensitive: bool=True, condition: Callable[[], bool]=None) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        ' Get objects that satisfy the given subject and predicate.\n\n        To understand the order of parameters, keep in ming the following generic construction:\n        (<subject_type>, <subject>, <predicate>, <object>).\n\n        So in the case of retrieving objects this construction becomes\n        (<subject_type>, <subject>, <predicate>, ?).\n\n        Args:\n            subject_type: a type of the subject.\n            subject: a string that represents the subject.\n            predicate: the enum that represents a predicate of querying operations.\n            case_sensitive: if True, then Resources are selected in a case-sensitive manner. if False, then Resources\n                are selected in a case-insensitive manner.\n\n        Returns: a list of the strings representing the objects.\n        '
        self.logger.debug(f'Get subjects for {subject} with {predicate}')
        statements = self._get_statements(source_type=subject_type, source_name=subject, statements_getter=lambda r: r.subject_statements, target_condition=(lambda s: s.object.type == predicate.value) if predicate else lambda _: True, condition=condition or self._show_condition, case_sensitive=case_sensitive)
        return [s.object.name for s in statements]

    def get_subjects(self, subject_type: Optional[ResourceType]=None, predicate: Optional[ResourceType]=None, obj: Optional[str]='', case_sensitive: bool=True) -> List[str]:
        if False:
            print('Hello World!')
        ' Get subjects that satisfy the given object and predicate.\n        To understand the order of parameters, keep in ming the following generic construction:\n\n        (<subject_type>, <subject>, <predicate>, <object>).\n\n        So in the case of retrieving subjects this construction becomes\n        (<subject_type>, ?, <predicate>, <object>).\n\n        Args:\n            subject_type: a type of the subject.\n            obj: a string that represents the object.\n            predicate: the enum that represents a predicate of querying operations.\n            case_sensitive: if True, then Resources are selected in a case-sensitive manner. if False, then Resources\n                are selected in a case-insensitive manner.\n\n        Returns: a list of the strings representing the subjects.\n        '
        self.logger.debug(f'Get linked back resources for {obj} with {predicate}')
        statements = self._get_statements(source_type=predicate, source_name=obj, statements_getter=lambda r: r.object_statements, target_condition=(lambda s: s.subject.type == subject_type.value) if subject_type else lambda _: True, condition=self._show_condition, case_sensitive=case_sensitive)
        return [s.subject.name for s in statements]

    def get_statements(self, subject_type: Optional[ResourceType]=None, subject: Optional[str]='', case_sensitive: bool=True) -> List[SimpleStatement]:
        if False:
            while True:
                i = 10
        statements = self._get_statements(source_type=subject_type, source_name=subject, statements_getter=lambda r: r.subject_statements, target_condition=lambda _: True, condition=self._show_condition, case_sensitive=case_sensitive)
        statements = map(lambda s: SimpleStatement(subject_type=s.subject.type, subject=s.subject.name, predicate=s.object.type, object=s.object.name), statements)
        return list(statements)

    def get_suggestions(self, subject_type: Optional[ResourceType]=None, subject: Optional[str]='', predicate: Optional[ResourceType]=None, case_sensitive: bool=True) -> List[str]:
        if False:
            print('Hello World!')
        ' Get all suggestions for a particular subject.\n\n        Args:\n            subject_type: a type of the subject.\n            subject: a string that represents the subject.\n            predicate: the enum that represents a predicate of querying operations.\n            case_sensitive: if True, then Resources are selected in a case-sensitive manner. if False, then Resources\n                are selected in a case-insensitive manner.\n\n        Returns: a list of the strings representing the objects.\n        '
        self.logger.debug(f'Getting suggestions for {subject} with {predicate}')
        suggestions = self.get_objects(subject_type=subject_type, subject=subject, predicate=predicate, case_sensitive=case_sensitive, condition=lambda s: not s.local_operation and between(s.score, HIDE_THRESHOLD + 1, SHOW_THRESHOLD - 1))
        return suggestions

    def get_subjects_intersection(self, objects: Set[str], predicate: Optional[ResourceType], subjects_type: Optional[ResourceType]=ResourceType.TORRENT, case_sensitive: bool=True) -> Set[str]:
        if False:
            return 10
        if not objects:
            return set()
        if case_sensitive:
            name_condition = '"obj"."name" = $obj_name'
        else:
            name_condition = 'py_lower("obj"."name") = py_lower($obj_name)'
        query = select((r.name for r in iterable(self.Resource) if r.type == subjects_type.value))
        for obj_name in objects:
            query = query.filter(raw_sql(f'\n    r.id IN (\n        SELECT "s"."subject"\n        FROM "Statement" "s"\n        WHERE (\n            "s"."local_operation" = $(Operation.ADD.value)\n        OR\n            ("s"."local_operation" = 0 OR "s"."local_operation" IS NULL)\n            AND ("s"."added_count" - "s"."removed_count") >= $SHOW_THRESHOLD\n        ) AND "s"."object" IN (\n            SELECT "obj"."id" FROM "Resource" "obj"\n            WHERE "obj"."type" = $(predicate.value) AND {name_condition}\n        )\n    )'))
        return set(query)

    def get_clock(self, operation: StatementOperation) -> int:
        if False:
            return 10
        ' Get the clock (int) of operation.\n        '
        peer = self.Peer.get(public_key=operation.creator_public_key)
        subject = self.Resource.get(name=operation.subject, type=operation.subject_type)
        obj = self.Resource.get(name=operation.object, type=operation.predicate)
        if not subject or not obj or (not peer):
            return CLOCK_START_VALUE
        statement = self.Statement.get(subject=subject, object=obj)
        if not statement:
            return CLOCK_START_VALUE
        op = self.StatementOp.get(statement=statement, peer=peer)
        return op.clock if op else CLOCK_START_VALUE

    def get_operations_for_gossip(self, count: int=10) -> Set[Entity]:
        if False:
            for i in range(10):
                print('nop')
        ' Get random operations from the DB that older than time_delta.\n\n        Args:\n            count: a limit for a resulting query\n        '
        return self._get_random_operations_by_condition(condition=lambda so: not so.auto_generated, count=count)

    def _get_random_operations_by_condition(self, condition: Callable[[Entity], bool], count: int=5, attempts: int=100) -> Set[Entity]:
        if False:
            while True:
                i = 10
        ' Get `count` random operations that satisfy the given condition.\n\n        This method were introduce as an fast alternative for native Pony `random` method.\n\n\n        Args:\n            condition: the condition by which the entities will be queried.\n            count: the amount of entities to return.\n            attempts: maximum attempt count for requesting the DB.\n\n        Returns: a set of random operations\n        '
        operations = set()
        for _ in range(attempts):
            if len(operations) == count:
                return operations
            random_operations_list = self.StatementOp.select_random(1)
            if random_operations_list:
                operation = random_operations_list[0]
                if condition(operation):
                    operations.add(operation)
        return operations