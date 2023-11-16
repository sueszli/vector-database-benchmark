"""This module contains Cloud Memorystore links."""
from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.providers.google.cloud.links.base import BaseGoogleLink
if TYPE_CHECKING:
    from airflow.models import BaseOperator
    from airflow.utils.context import Context
BASE_LINK = '/memorystore'
MEMCACHED_LINK = BASE_LINK + '/memcached/locations/{location_id}/instances/{instance_id}/details?project={project_id}'
MEMCACHED_LIST_LINK = BASE_LINK + '/memcached/instances?project={project_id}'
REDIS_LINK = BASE_LINK + '/redis/locations/{location_id}/instances/{instance_id}/details/overview?project={project_id}'
REDIS_LIST_LINK = BASE_LINK + '/redis/instances?project={project_id}'

class MemcachedInstanceDetailsLink(BaseGoogleLink):
    """Helper class for constructing Memorystore Memcached Instance Link."""
    name = 'Memorystore Memcached Instance'
    key = 'memcached_instance'
    format_str = MEMCACHED_LINK

    @staticmethod
    def persist(context: Context, task_instance: BaseOperator, instance_id: str, location_id: str, project_id: str | None):
        if False:
            return 10
        task_instance.xcom_push(context, key=MemcachedInstanceDetailsLink.key, value={'instance_id': instance_id, 'location_id': location_id, 'project_id': project_id})

class MemcachedInstanceListLink(BaseGoogleLink):
    """Helper class for constructing Memorystore Memcached List of Instances Link."""
    name = 'Memorystore Memcached List of Instances'
    key = 'memcached_instances'
    format_str = MEMCACHED_LIST_LINK

    @staticmethod
    def persist(context: Context, task_instance: BaseOperator, project_id: str | None):
        if False:
            i = 10
            return i + 15
        task_instance.xcom_push(context, key=MemcachedInstanceListLink.key, value={'project_id': project_id})

class RedisInstanceDetailsLink(BaseGoogleLink):
    """Helper class for constructing Memorystore Redis Instance Link."""
    name = 'Memorystore Redis Instance'
    key = 'redis_instance'
    format_str = REDIS_LINK

    @staticmethod
    def persist(context: Context, task_instance: BaseOperator, instance_id: str, location_id: str, project_id: str | None):
        if False:
            return 10
        task_instance.xcom_push(context, key=RedisInstanceDetailsLink.key, value={'instance_id': instance_id, 'location_id': location_id, 'project_id': project_id})

class RedisInstanceListLink(BaseGoogleLink):
    """Helper class for constructing Memorystore Redis List of Instances Link."""
    name = 'Memorystore Redis List of Instances'
    key = 'redis_instances'
    format_str = REDIS_LIST_LINK

    @staticmethod
    def persist(context: Context, task_instance: BaseOperator, project_id: str | None):
        if False:
            for i in range(10):
                print('nop')
        task_instance.xcom_push(context, key=RedisInstanceListLink.key, value={'project_id': project_id})