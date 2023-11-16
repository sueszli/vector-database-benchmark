# pylint: disable=too-many-lines
# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import sys
from typing import Any, AsyncIterable, Callable, Dict, IO, Optional, TypeVar, Union, overload
import urllib.parse

from azure.core.async_paging import AsyncItemPaged, AsyncList
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    ResourceNotModifiedError,
    map_error,
)
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import AsyncHttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.tracing.decorator_async import distributed_trace_async
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat

from ... import models as _models
from ..._vendor import _convert_request
from ...operations._service_tasks_operations import (
    build_cancel_request,
    build_create_or_update_request,
    build_delete_request,
    build_get_request,
    build_list_request,
    build_update_request,
)

if sys.version_info >= (3, 8):
    from typing import Literal  # pylint: disable=no-name-in-module, ungrouped-imports
else:
    from typing_extensions import Literal  # type: ignore  # pylint: disable=ungrouped-imports
T = TypeVar("T")
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]


class ServiceTasksOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.datamigration.aio.DataMigrationManagementClient`'s
        :attr:`service_tasks` attribute.
    """

    models = _models

    def __init__(self, *args, **kwargs) -> None:
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop("client")
        self._config = input_args.pop(0) if input_args else kwargs.pop("config")
        self._serialize = input_args.pop(0) if input_args else kwargs.pop("serializer")
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop("deserializer")

    @distributed_trace
    def list(
        self, group_name: str, service_name: str, task_type: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterable["_models.ProjectTask"]:
        """Get service level tasks for a service.

        The services resource is the top-level resource that represents the Database Migration Service.
        This method returns a list of service level tasks owned by a service resource. Some tasks may
        have a status of Unknown, which indicates that an error occurred while querying the status of
        that task.

        :param group_name: Name of the resource group. Required.
        :type group_name: str
        :param service_name: Name of the service. Required.
        :type service_name: str
        :param task_type: Filter tasks by task type. Default value is None.
        :type task_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: An iterator like instance of either ProjectTask or the result of cls(response)
        :rtype: ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.datamigration.models.ProjectTask]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2022-03-30-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", self._config.api_version)
        )
        cls: ClsType[_models.TaskList] = kwargs.pop("cls", None)

        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        def prepare_request(next_link=None):
            if not next_link:

                request = build_list_request(
                    group_name=group_name,
                    service_name=service_name,
                    subscription_id=self._config.subscription_id,
                    task_type=task_type,
                    api_version=api_version,
                    template_url=self.list.metadata["url"],
                    headers=_headers,
                    params=_params,
                )
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)

            else:
                # make call to next link with the client's api-version
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict(
                    {
                        key: [urllib.parse.quote(v) for v in value]
                        for key, value in urllib.parse.parse_qs(_parsed_next_link.query).items()
                    }
                )
                _next_request_params["api-version"] = self._config.api_version
                request = HttpRequest(
                    "GET", urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params
                )
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = "GET"
            return request

        async def extract_data(pipeline_response):
            deserialized = self._deserialize("TaskList", pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)  # type: ignore
            return deserialized.next_link or None, AsyncList(list_of_elem)

        async def get_next(next_link=None):
            request = prepare_request(next_link)

            pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
                request, stream=False, **kwargs
            )
            response = pipeline_response.http_response

            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.ApiError, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)

            return pipeline_response

        return AsyncItemPaged(get_next, extract_data)

    list.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{groupName}/providers/Microsoft.DataMigration/services/{serviceName}/serviceTasks"
    }

    @overload
    async def create_or_update(
        self,
        group_name: str,
        service_name: str,
        task_name: str,
        parameters: _models.ProjectTask,
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> _models.ProjectTask:
        """Create or update service task.

        The service tasks resource is a nested, proxy-only resource representing work performed by a
        DMS instance. The PUT method creates a new service task or updates an existing one, although
        since service tasks have no mutable custom properties, there is little reason to update an
        existing one.

        :param group_name: Name of the resource group. Required.
        :type group_name: str
        :param service_name: Name of the service. Required.
        :type service_name: str
        :param task_name: Name of the Task. Required.
        :type task_name: str
        :param parameters: Information about the task. Required.
        :type parameters: ~azure.mgmt.datamigration.models.ProjectTask
        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.
         Default value is "application/json".
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: ProjectTask or the result of cls(response)
        :rtype: ~azure.mgmt.datamigration.models.ProjectTask
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @overload
    async def create_or_update(
        self,
        group_name: str,
        service_name: str,
        task_name: str,
        parameters: IO,
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> _models.ProjectTask:
        """Create or update service task.

        The service tasks resource is a nested, proxy-only resource representing work performed by a
        DMS instance. The PUT method creates a new service task or updates an existing one, although
        since service tasks have no mutable custom properties, there is little reason to update an
        existing one.

        :param group_name: Name of the resource group. Required.
        :type group_name: str
        :param service_name: Name of the service. Required.
        :type service_name: str
        :param task_name: Name of the Task. Required.
        :type task_name: str
        :param parameters: Information about the task. Required.
        :type parameters: IO
        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.
         Default value is "application/json".
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: ProjectTask or the result of cls(response)
        :rtype: ~azure.mgmt.datamigration.models.ProjectTask
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @distributed_trace_async
    async def create_or_update(
        self,
        group_name: str,
        service_name: str,
        task_name: str,
        parameters: Union[_models.ProjectTask, IO],
        **kwargs: Any
    ) -> _models.ProjectTask:
        """Create or update service task.

        The service tasks resource is a nested, proxy-only resource representing work performed by a
        DMS instance. The PUT method creates a new service task or updates an existing one, although
        since service tasks have no mutable custom properties, there is little reason to update an
        existing one.

        :param group_name: Name of the resource group. Required.
        :type group_name: str
        :param service_name: Name of the service. Required.
        :type service_name: str
        :param task_name: Name of the Task. Required.
        :type task_name: str
        :param parameters: Information about the task. Is either a model type or a IO type. Required.
        :type parameters: ~azure.mgmt.datamigration.models.ProjectTask or IO
        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.
         Default value is None.
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: ProjectTask or the result of cls(response)
        :rtype: ~azure.mgmt.datamigration.models.ProjectTask
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2022-03-30-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", self._config.api_version)
        )
        content_type: Optional[str] = kwargs.pop("content_type", _headers.pop("Content-Type", None))
        cls: ClsType[_models.ProjectTask] = kwargs.pop("cls", None)

        content_type = content_type or "application/json"
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, "ProjectTask")

        request = build_create_or_update_request(
            group_name=group_name,
            service_name=service_name,
            task_name=task_name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            content_type=content_type,
            json=_json,
            content=_content,
            template_url=self.create_or_update.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            request, stream=False, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ApiError, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)

        if response.status_code == 200:
            deserialized = self._deserialize("ProjectTask", pipeline_response)

        if response.status_code == 201:
            deserialized = self._deserialize("ProjectTask", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore

    create_or_update.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{groupName}/providers/Microsoft.DataMigration/services/{serviceName}/serviceTasks/{taskName}"
    }

    @distributed_trace_async
    async def get(
        self, group_name: str, service_name: str, task_name: str, expand: Optional[str] = None, **kwargs: Any
    ) -> _models.ProjectTask:
        """Get service task information.

        The service tasks resource is a nested, proxy-only resource representing work performed by a
        DMS instance. The GET method retrieves information about a service task.

        :param group_name: Name of the resource group. Required.
        :type group_name: str
        :param service_name: Name of the service. Required.
        :type service_name: str
        :param task_name: Name of the Task. Required.
        :type task_name: str
        :param expand: Expand the response. Default value is None.
        :type expand: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: ProjectTask or the result of cls(response)
        :rtype: ~azure.mgmt.datamigration.models.ProjectTask
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2022-03-30-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", self._config.api_version)
        )
        cls: ClsType[_models.ProjectTask] = kwargs.pop("cls", None)

        request = build_get_request(
            group_name=group_name,
            service_name=service_name,
            task_name=task_name,
            subscription_id=self._config.subscription_id,
            expand=expand,
            api_version=api_version,
            template_url=self.get.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            request, stream=False, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ApiError, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)

        deserialized = self._deserialize("ProjectTask", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    get.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{groupName}/providers/Microsoft.DataMigration/services/{serviceName}/serviceTasks/{taskName}"
    }

    @distributed_trace_async
    async def delete(  # pylint: disable=inconsistent-return-statements
        self,
        group_name: str,
        service_name: str,
        task_name: str,
        delete_running_tasks: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        """Delete service task.

        The service tasks resource is a nested, proxy-only resource representing work performed by a
        DMS instance. The DELETE method deletes a service task, canceling it first if it's running.

        :param group_name: Name of the resource group. Required.
        :type group_name: str
        :param service_name: Name of the service. Required.
        :type service_name: str
        :param task_name: Name of the Task. Required.
        :type task_name: str
        :param delete_running_tasks: Delete the resource even if it contains running tasks. Default
         value is None.
        :type delete_running_tasks: bool
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: None or the result of cls(response)
        :rtype: None
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2022-03-30-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", self._config.api_version)
        )
        cls: ClsType[None] = kwargs.pop("cls", None)

        request = build_delete_request(
            group_name=group_name,
            service_name=service_name,
            task_name=task_name,
            subscription_id=self._config.subscription_id,
            delete_running_tasks=delete_running_tasks,
            api_version=api_version,
            template_url=self.delete.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            request, stream=False, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ApiError, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)

        if cls:
            return cls(pipeline_response, None, {})

    delete.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{groupName}/providers/Microsoft.DataMigration/services/{serviceName}/serviceTasks/{taskName}"
    }

    @overload
    async def update(
        self,
        group_name: str,
        service_name: str,
        task_name: str,
        parameters: _models.ProjectTask,
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> _models.ProjectTask:
        """Create or update service task.

        The service tasks resource is a nested, proxy-only resource representing work performed by a
        DMS instance. The PATCH method updates an existing service task, but since service tasks have
        no mutable custom properties, there is little reason to do so.

        :param group_name: Name of the resource group. Required.
        :type group_name: str
        :param service_name: Name of the service. Required.
        :type service_name: str
        :param task_name: Name of the Task. Required.
        :type task_name: str
        :param parameters: Information about the task. Required.
        :type parameters: ~azure.mgmt.datamigration.models.ProjectTask
        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.
         Default value is "application/json".
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: ProjectTask or the result of cls(response)
        :rtype: ~azure.mgmt.datamigration.models.ProjectTask
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @overload
    async def update(
        self,
        group_name: str,
        service_name: str,
        task_name: str,
        parameters: IO,
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> _models.ProjectTask:
        """Create or update service task.

        The service tasks resource is a nested, proxy-only resource representing work performed by a
        DMS instance. The PATCH method updates an existing service task, but since service tasks have
        no mutable custom properties, there is little reason to do so.

        :param group_name: Name of the resource group. Required.
        :type group_name: str
        :param service_name: Name of the service. Required.
        :type service_name: str
        :param task_name: Name of the Task. Required.
        :type task_name: str
        :param parameters: Information about the task. Required.
        :type parameters: IO
        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.
         Default value is "application/json".
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: ProjectTask or the result of cls(response)
        :rtype: ~azure.mgmt.datamigration.models.ProjectTask
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @distributed_trace_async
    async def update(
        self,
        group_name: str,
        service_name: str,
        task_name: str,
        parameters: Union[_models.ProjectTask, IO],
        **kwargs: Any
    ) -> _models.ProjectTask:
        """Create or update service task.

        The service tasks resource is a nested, proxy-only resource representing work performed by a
        DMS instance. The PATCH method updates an existing service task, but since service tasks have
        no mutable custom properties, there is little reason to do so.

        :param group_name: Name of the resource group. Required.
        :type group_name: str
        :param service_name: Name of the service. Required.
        :type service_name: str
        :param task_name: Name of the Task. Required.
        :type task_name: str
        :param parameters: Information about the task. Is either a model type or a IO type. Required.
        :type parameters: ~azure.mgmt.datamigration.models.ProjectTask or IO
        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.
         Default value is None.
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: ProjectTask or the result of cls(response)
        :rtype: ~azure.mgmt.datamigration.models.ProjectTask
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2022-03-30-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", self._config.api_version)
        )
        content_type: Optional[str] = kwargs.pop("content_type", _headers.pop("Content-Type", None))
        cls: ClsType[_models.ProjectTask] = kwargs.pop("cls", None)

        content_type = content_type or "application/json"
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, "ProjectTask")

        request = build_update_request(
            group_name=group_name,
            service_name=service_name,
            task_name=task_name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            content_type=content_type,
            json=_json,
            content=_content,
            template_url=self.update.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            request, stream=False, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ApiError, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)

        deserialized = self._deserialize("ProjectTask", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    update.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{groupName}/providers/Microsoft.DataMigration/services/{serviceName}/serviceTasks/{taskName}"
    }

    @distributed_trace_async
    async def cancel(self, group_name: str, service_name: str, task_name: str, **kwargs: Any) -> _models.ProjectTask:
        """Cancel a service task.

        The service tasks resource is a nested, proxy-only resource representing work performed by a
        DMS instance. This method cancels a service task if it's currently queued or running.

        :param group_name: Name of the resource group. Required.
        :type group_name: str
        :param service_name: Name of the service. Required.
        :type service_name: str
        :param task_name: Name of the Task. Required.
        :type task_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: ProjectTask or the result of cls(response)
        :rtype: ~azure.mgmt.datamigration.models.ProjectTask
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2022-03-30-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", self._config.api_version)
        )
        cls: ClsType[_models.ProjectTask] = kwargs.pop("cls", None)

        request = build_cancel_request(
            group_name=group_name,
            service_name=service_name,
            task_name=task_name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            template_url=self.cancel.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            request, stream=False, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ApiError, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)

        deserialized = self._deserialize("ProjectTask", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    cancel.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{groupName}/providers/Microsoft.DataMigration/services/{serviceName}/serviceTasks/{taskName}/cancel"
    }
