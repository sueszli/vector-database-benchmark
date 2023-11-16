import logging
import typing
from contextlib import suppress
from typing import Iterable

import boto3
from boto3.dynamodb.conditions import Key
from botocore.config import Config
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from flag_engine.environments.builders import build_environment_model
from flag_engine.identities.builders import build_identity_model
from flag_engine.identities.models import IdentityModel
from flag_engine.segments.evaluator import get_identity_segments
from rest_framework.exceptions import NotFound

from util.mappers import (
    map_environment_api_key_to_environment_api_key_document,
    map_environment_to_environment_document,
    map_identity_to_identity_document,
)

if typing.TYPE_CHECKING:
    from environments.identities.models import Identity
    from environments.models import Environment, EnvironmentAPIKey

logger = logging.getLogger()


class DynamoWrapper:
    table_name: str = None

    def __init__(self):
        self._table = None
        if self.table_name:
            self._table = boto3.resource(
                "dynamodb", config=Config(tcp_keepalive=True)
            ).Table(self.table_name)

    @property
    def is_enabled(self) -> bool:
        return self._table is not None


class DynamoIdentityWrapper(DynamoWrapper):
    table_name = settings.IDENTITIES_TABLE_NAME_DYNAMO

    def query_items(self, *args, **kwargs):
        return self._table.query(*args, **kwargs)

    def put_item(self, identity_dict: dict):
        self._table.put_item(Item=identity_dict)

    def write_identities(self, identities: Iterable["Identity"]):
        with self._table.batch_writer() as batch:
            for identity in identities:
                identity_document = map_identity_to_identity_document(identity)
                # Since sort keys can not be greater than 1024
                # https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ServiceQuotas.html#limits-partition-sort-keys
                if len(identity_document["identifier"]) > 1024:
                    logger.warning(
                        f"Can't migrate identity {identity.id}; identifier too long"
                    )
                    continue
                batch.put_item(Item=identity_document)

    def get_item(self, composite_key: str) -> typing.Optional[dict]:
        return self._table.get_item(Key={"composite_key": composite_key}).get("Item")

    def delete_item(self, composite_key: str):
        self._table.delete_item(Key={"composite_key": composite_key})

    def get_item_from_uuid(self, uuid: str) -> dict:
        filter_expression = Key("identity_uuid").eq(uuid)
        query_kwargs = {
            "IndexName": "identity_uuid-index",
            "Limit": 1,
            "KeyConditionExpression": filter_expression,
        }
        try:
            return self.query_items(**query_kwargs)["Items"][0]
        except IndexError:
            raise ObjectDoesNotExist()

    def get_item_from_uuid_or_404(self, uuid: str) -> dict:
        try:
            return self.get_item_from_uuid(uuid)
        except ObjectDoesNotExist as e:
            raise NotFound() from e

    def get_all_items(
        self, environment_api_key: str, limit: int, start_key: dict = None
    ):
        filter_expression = Key("environment_api_key").eq(environment_api_key)
        query_kwargs = {
            "IndexName": "environment_api_key-identifier-index",
            "Limit": limit,
            "KeyConditionExpression": filter_expression,
        }
        if start_key:
            query_kwargs.update(ExclusiveStartKey=start_key)
        return self.query_items(**query_kwargs)

    def search_items_with_identifier(
        self,
        environment_api_key: str,
        identifier: str,
        search_function: typing.Callable,
        limit: int,
        start_key: dict = None,
    ):
        filter_expression = Key("environment_api_key").eq(
            environment_api_key
        ) & search_function(identifier)
        query_kwargs = {
            "IndexName": "environment_api_key-identifier-index",
            "Limit": limit,
            "KeyConditionExpression": filter_expression,
        }
        if start_key:
            query_kwargs.update(ExclusiveStartKey=start_key)
        return self.query_items(**query_kwargs)

    def get_segment_ids(
        self, identity_pk: str = None, identity_model: IdentityModel = None
    ) -> list:
        if not (identity_pk or identity_model):
            raise ValueError("Must provide one of identity_pk or identity_model.")

        with suppress(ObjectDoesNotExist):
            identity = identity_model or build_identity_model(
                self.get_item_from_uuid(identity_pk)
            )
            environment_wrapper = DynamoEnvironmentWrapper()
            environment = build_environment_model(
                environment_wrapper.get_item(identity.environment_api_key)
            )
            segments = get_identity_segments(environment, identity)
            return [segment.id for segment in segments]

        return []


class DynamoEnvironmentWrapper(DynamoWrapper):
    table_name = settings.ENVIRONMENTS_TABLE_NAME_DYNAMO

    def write_environment(self, environment: "Environment"):
        self.write_environments([environment])

    def write_environments(self, environments: Iterable["Environment"]):
        with self._table.batch_writer() as writer:
            for environment in environments:
                writer.put_item(
                    Item=map_environment_to_environment_document(environment),
                )

    def get_item(self, api_key: str) -> dict:
        try:
            return self._table.get_item(Key={"api_key": api_key})["Item"]
        except KeyError as e:
            raise ObjectDoesNotExist() from e


class DynamoEnvironmentAPIKeyWrapper(DynamoWrapper):
    table_name = settings.ENVIRONMENTS_API_KEY_TABLE_NAME_DYNAMO

    def write_api_key(self, api_key: "EnvironmentAPIKey"):
        self.write_api_keys([api_key])

    def write_api_keys(self, api_keys: Iterable["EnvironmentAPIKey"]):
        with self._table.batch_writer() as writer:
            for api_key in api_keys:
                writer.put_item(
                    Item=map_environment_api_key_to_environment_api_key_document(
                        api_key
                    )
                )
