"""Connect to Asana."""
from __future__ import annotations
from functools import cached_property
from typing import Any
from asana import Client
from asana.error import NotFoundError
from airflow.hooks.base import BaseHook

class AsanaHook(BaseHook):
    """Wrapper around Asana Python client library."""
    conn_name_attr = 'asana_conn_id'
    default_conn_name = 'asana_default'
    conn_type = 'asana'
    hook_name = 'Asana'

    def __init__(self, conn_id: str=default_conn_name, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.connection = self.get_connection(conn_id)
        extras = self.connection.extra_dejson
        self.workspace = self._get_field(extras, 'workspace') or None
        self.project = self._get_field(extras, 'project') or None

    def _get_field(self, extras: dict, field_name: str):
        if False:
            print('Hello World!')
        'Get field from extra, first checking short name, then for backcompat we check for prefixed name.'
        backcompat_prefix = 'extra__asana__'
        if field_name.startswith('extra__'):
            raise ValueError(f"Got prefixed name {field_name}; please remove the '{backcompat_prefix}' prefix when using this method.")
        if field_name in extras:
            return extras[field_name] or None
        prefixed_name = f'{backcompat_prefix}{field_name}'
        return extras.get(prefixed_name) or None

    def get_conn(self) -> Client:
        if False:
            return 10
        return self.client

    @staticmethod
    def get_connection_form_widgets() -> dict[str, Any]:
        if False:
            print('Hello World!')
        'Return connection widgets to add to connection form.'
        from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
        from flask_babel import lazy_gettext
        from wtforms import StringField
        return {'workspace': StringField(lazy_gettext('Workspace'), widget=BS3TextFieldWidget()), 'project': StringField(lazy_gettext('Project'), widget=BS3TextFieldWidget())}

    @staticmethod
    def get_ui_field_behaviour() -> dict[str, Any]:
        if False:
            while True:
                i = 10
        'Return custom field behaviour.'
        return {'hidden_fields': ['port', 'host', 'login', 'schema'], 'relabeling': {}, 'placeholders': {'password': 'Asana personal access token', 'workspace': 'Asana workspace gid', 'project': 'Asana project gid'}}

    @cached_property
    def client(self) -> Client:
        if False:
            i = 10
            return i + 15
        'Instantiate python-asana Client.'
        if not self.connection.password:
            raise ValueError('Asana connection password must contain a personal access token: https://developers.asana.com/docs/personal-access-token')
        return Client.access_token(self.connection.password)

    def create_task(self, task_name: str, params: dict | None) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Create an Asana task.\n\n        :param task_name: Name of the new task\n        :param params: Other task attributes, such as due_on, parent, and notes. For a complete list\n            of possible parameters, see https://developers.asana.com/docs/create-a-task\n        :return: A dict of attributes of the created task, including its gid\n        '
        merged_params = self._merge_create_task_parameters(task_name, params)
        self._validate_create_task_parameters(merged_params)
        response = self.client.tasks.create(params=merged_params)
        return response

    def _merge_create_task_parameters(self, task_name: str, task_params: dict | None) -> dict:
        if False:
            while True:
                i = 10
        '\n        Merge create_task parameters with default params from the connection.\n\n        :param task_name: Name of the task\n        :param task_params: Other task parameters which should override defaults from the connection\n        :return: A dict of merged parameters to use in the new task\n        '
        merged_params: dict[str, Any] = {'name': task_name}
        if self.project:
            merged_params['projects'] = [self.project]
        elif self.workspace and (not (task_params and 'projects' in task_params)):
            merged_params['workspace'] = self.workspace
        if task_params:
            merged_params.update(task_params)
        return merged_params

    @staticmethod
    def _validate_create_task_parameters(params: dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Check that user provided minimal parameters for task creation.\n\n        :param params: A dict of attributes the task to be created should have\n        :return: None; raises ValueError if `params` doesn't contain required parameters\n        "
        required_parameters = {'workspace', 'projects', 'parent'}
        if required_parameters.isdisjoint(params):
            raise ValueError(f'You must specify at least one of {required_parameters} in the create_task parameters')

    def delete_task(self, task_id: str) -> dict:
        if False:
            while True:
                i = 10
        '\n        Delete an Asana task.\n\n        :param task_id: Asana GID of the task to delete\n        :return: A dict containing the response from Asana\n        '
        try:
            response = self.client.tasks.delete_task(task_id)
            return response
        except NotFoundError:
            self.log.info('Asana task %s not found for deletion.', task_id)
            return {}

    def find_task(self, params: dict | None) -> list:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve a list of Asana tasks that match search parameters.\n\n        :param params: Attributes that matching tasks should have. For a list of possible parameters,\n            see https://developers.asana.com/docs/get-multiple-tasks\n        :return: A list of dicts containing attributes of matching Asana tasks\n        '
        merged_params = self._merge_find_task_parameters(params)
        self._validate_find_task_parameters(merged_params)
        response = self.client.tasks.find_all(params=merged_params)
        return list(response)

    def _merge_find_task_parameters(self, search_parameters: dict | None) -> dict:
        if False:
            return 10
        '\n        Merge find_task parameters with default params from the connection.\n\n        :param search_parameters: Attributes that tasks matching the search should have; these override\n            defaults from the connection\n        :return: A dict of merged parameters to use in the search\n        '
        merged_params = {}
        if self.project:
            merged_params['project'] = self.project
        elif self.workspace and (not (search_parameters and 'project' in search_parameters)):
            merged_params['workspace'] = self.workspace
        if search_parameters:
            merged_params.update(search_parameters)
        return merged_params

    @staticmethod
    def _validate_find_task_parameters(params: dict) -> None:
        if False:
            while True:
                i = 10
        '\n        Check that the user provided minimal search parameters.\n\n        :param params: Dict of parameters to be used in the search\n        :return: None; raises ValueError if search parameters do not contain minimum required attributes\n        '
        one_of_list = {'project', 'section', 'tag', 'user_task_list'}
        both_of_list = {'assignee', 'workspace'}
        contains_both = both_of_list.issubset(params)
        contains_one = not one_of_list.isdisjoint(params)
        if not (contains_both or contains_one):
            raise ValueError(f'You must specify at least one of {one_of_list} or both of {both_of_list} in the find_task parameters.')

    def update_task(self, task_id: str, params: dict) -> dict:
        if False:
            print('Hello World!')
        "\n        Update an existing Asana task.\n\n        :param task_id: Asana GID of task to update\n        :param params: New values of the task's attributes. For a list of possible parameters, see\n            https://developers.asana.com/docs/update-a-task\n        :return: A dict containing the updated task's attributes\n        "
        response = self.client.tasks.update(task_id, params)
        return response

    def create_project(self, params: dict) -> dict:
        if False:
            while True:
                i = 10
        "\n        Create a new project.\n\n        :param params: Attributes that the new project should have. See\n            https://developers.asana.com/docs/create-a-project#create-a-project-parameters\n            for a list of possible parameters.\n        :return: A dict containing the new project's attributes, including its GID.\n        "
        merged_params = self._merge_project_parameters(params)
        self._validate_create_project_parameters(merged_params)
        response = self.client.projects.create(merged_params)
        return response

    @staticmethod
    def _validate_create_project_parameters(params: dict) -> None:
        if False:
            return 10
        '\n        Check that user provided the minimum required parameters for project creation.\n\n        :param params: Attributes that the new project should have\n        :return: None; raises a ValueError if `params` does not contain the minimum required attributes.\n        '
        required_parameters = {'workspace', 'team'}
        if required_parameters.isdisjoint(params):
            raise ValueError(f'You must specify at least one of {required_parameters} in the create_project params')

    def _merge_project_parameters(self, params: dict) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Merge parameters passed into a project method with default params from the connection.\n\n        :param params: Parameters passed into one of the project methods, which should override\n            defaults from the connection\n        :return: A dict of merged parameters\n        '
        merged_params = {} if self.workspace is None else {'workspace': self.workspace}
        merged_params.update(params)
        return merged_params

    def find_project(self, params: dict) -> list:
        if False:
            while True:
                i = 10
        '\n        Retrieve a list of Asana projects that match search parameters.\n\n        :param params: Attributes which matching projects should have. See\n            https://developers.asana.com/docs/get-multiple-projects\n            for a list of possible parameters.\n        :return: A list of dicts containing attributes of matching Asana projects\n        '
        merged_params = self._merge_project_parameters(params)
        response = self.client.projects.find_all(merged_params)
        return list(response)

    def update_project(self, project_id: str, params: dict) -> dict:
        if False:
            for i in range(10):
                print('nop')
        "\n        Update an existing project.\n\n        :param project_id: Asana GID of the project to update\n        :param params: New attributes that the project should have. See\n            https://developers.asana.com/docs/update-a-project#update-a-project-parameters\n            for a list of possible parameters\n        :return: A dict containing the updated project's attributes\n        "
        response = self.client.projects.update(project_id, params)
        return response

    def delete_project(self, project_id: str) -> dict:
        if False:
            return 10
        '\n        Delete a project.\n\n        :param project_id: Asana GID of the project to delete\n        :return: A dict containing the response from Asana\n        '
        try:
            response = self.client.projects.delete(project_id)
            return response
        except NotFoundError:
            self.log.info('Asana project %s not found for deletion.', project_id)
            return {}