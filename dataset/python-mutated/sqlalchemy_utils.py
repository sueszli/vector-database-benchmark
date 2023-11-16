"""Utilities for interacting with SQLAlchemy types.

This module mostly deals with detecting which entities are involved in a query
that is about to be executed.

We must detect all entities properly to apply authorization.
"""
import sqlalchemy
from sqlalchemy import inspect
from sqlalchemy.orm.util import AliasedClass, AliasedInsp

def to_class(entity):
    if False:
        i = 10
        return i + 15
    'Get mapped class from SQLAlchemy entity.'
    if isinstance(entity, AliasedClass):
        return inspect(entity).class_
    elif inspect(entity, False) is not None:
        return inspect(entity).class_
    else:
        return entity
try:

    def all_entities_in_statement(statement):
        if False:
            while True:
                i = 10
        '\n        Get all ORM entities that will be loaded in a select statement.\n\n        The includes entities that will be loaded eagerly through relationships either specified in\n        the query options or as default loader strategies on the model definition.\n\n        https://docs.sqlalchemy.org/en/14/orm/loading_relationships.html#relationship-loading-with-loader-options\n        '
        entities = get_column_entities(statement)
        entities |= get_joinedload_entities(statement)
        entities |= default_load_entities(entities)
        return set(map(to_class, entities))

    def get_column_entities(statement):
        if False:
            print('Hello World!')
        'Get entities in statement that are referenced as columns.\n\n        Examples::\n\n            >> get_column_entities(query(A)) == {A}\n            >> get_column_entities(query(A.field)) == {A}\n            >> get_column_entities(query(A, B)) == {A, B})\n\n        Does not include eager loaded entities.\n        '

        def _entities_in_statement(statement):
            if False:
                i = 10
                return i + 15
            try:
                entities = (cd.get('entity') for cd in statement.column_descriptions)
                return {e for e in entities if e is not None}
            except AttributeError:
                return set()
        entities = _entities_in_statement(statement)
        for child in statement.get_children():
            entities |= get_column_entities(child)
        return entities

    def default_load_entities(entities, seen_relationships=None):
        if False:
            i = 10
            return i + 15
        'Find related entities that will be loaded on all queries to ``entities``\n           due to the default loader strategy.\n\n        For example::\n\n            class A(Base):\n                bs = relationship(B, lazy="joined")\n\n        The relationship ``bs`` would be loaded eagerly whenever ``A`` is queried because\n        `lazy="joined"`.\n\n        :param entities: The entities to lookup default load entities for.\n        '
        default_entities = set()
        for entity in entities:
            mapper = sqlalchemy.inspect(entity)
            if isinstance(mapper, AliasedInsp):
                mapper = mapper.mapper
            relationships = mapper.relationships
            if seen_relationships is None:
                seen_relationships = set()
            for rel in relationships.values():
                if rel in seen_relationships:
                    continue
                seen_relationships.add(rel)
                if rel.lazy == 'joined':
                    default_entities |= default_load_entities([rel.mapper], seen_relationships)
                    default_entities.add(rel.mapper)
        return default_entities

    def get_joinedload_entities(stmt):
        if False:
            for i in range(10):
                print('nop')
        'Get extra entities that are loaded from a ``stmt`` due to joinedload\n        options specified in the statement options.\n\n        These entities will not be returned directly by the query, but will prepopulate\n        relationships in the returned data.\n\n        For example::\n\n            get_joinedload_entities(query(A).options(joinedload(A.bs))) == {A, B}\n        '
        entities = set()
        for opt in stmt._with_options:
            if hasattr(opt, '_to_bind'):
                for b in opt._to_bind:
                    if ('lazy', 'joined') in b.strategy:
                        entities.add(b.path[-1].entity)
            elif hasattr(opt, 'context'):
                for (key, loadopt) in opt.context.items():
                    if key[0] == 'loader' and ('lazy', 'joined') in loadopt.strategy:
                        entities.add(key[1][-1].entity)
        return entities
except ImportError:

    def all_entities_in_statement(statement):
        if False:
            print('Hello World!')
        raise NotImplementedError('Unsupported on SQLAlchemy < 1.4')