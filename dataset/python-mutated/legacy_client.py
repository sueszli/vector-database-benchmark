from asyncio import Future
"A Python module for interacting with Slack's Web API."
import json
import os
import warnings
from io import IOBase
from typing import Union, Sequence, Optional, Dict, Tuple, Any, List
import slack_sdk.errors as e
from slack_sdk.models.views import View
from .legacy_base_client import LegacyBaseClient, SlackResponse
from .internal_utils import _parse_web_class_objects, _update_call_participants, _warn_if_text_or_attachment_fallback_is_missing, _remove_none_values, _to_v2_file_upload_item, _upload_file_via_v2_url, _validate_for_legacy_client, _print_files_upload_v2_suggestion
from ..models.attachments import Attachment
from ..models.blocks import Block
from ..models.metadata import Metadata

class LegacyWebClient(LegacyBaseClient):
    """A WebClient allows apps to communicate with the Slack Platform's Web API.

    https://api.slack.com/methods

    The Slack Web API is an interface for querying information from
    and enacting change in a Slack workspace.

    This client handles constructing and sending HTTP requests to Slack
    as well as parsing any responses received into a `SlackResponse`.

    Attributes:
        token (str): A string specifying an `xoxp-*` or `xoxb-*` token.
        base_url (str): A string representing the Slack API base URL.
            Default is `'https://www.slack.com/api/'`
        timeout (int): The maximum number of seconds the client will wait
            to connect and receive a response from Slack.
            Default is 30 seconds.
        ssl (SSLContext): An [`ssl.SSLContext`][1] instance, helpful for specifying
            your own custom certificate chain.
        proxy (str): String representing a fully-qualified URL to a proxy through
            which to route all requests to the Slack API. Even if this parameter
            is not specified, if any of the following environment variables are
            present, they will be loaded into this parameter: `HTTPS_PROXY`,
            `https_proxy`, `HTTP_PROXY` or `http_proxy`.
        headers (dict): Additional request headers to attach to all requests.

    Methods:
        `api_call`: Constructs a request and executes the API call to Slack.

    Example of recommended usage:
    ```python
        import os
        from slack_sdk.web.legacy_client import LegacyWebClient

        client = LegacyWebClient(token=os.environ['SLACK_API_TOKEN'])
        response = client.chat_postMessage(
            channel='#random',
            text="Hello world!")
        assert response["ok"]
        assert response["message"]["text"] == "Hello world!"
    ```

    Example manually creating an API request:
    ```python
        import os
        from slack_sdk.web.legacy_client import LegacyWebClient

        client = LegacyWebClient(token=os.environ['SLACK_API_TOKEN'])
        response = client.api_call(
            api_method='chat.postMessage',
            json={'channel': '#random','text': "Hello world!"}
        )
        assert response["ok"]
        assert response["message"]["text"] == "Hello world!"
    ```

    Note:
        Any attributes or methods prefixed with _underscores are
        intended to be "private" internal use only. They may be changed or
        removed at anytime.

    [1]: https://docs.python.org/3/library/ssl.html#ssl.SSLContext
    """

    def admin_analytics_getFile(self, *, type: str, date: Optional[str]=None, metadata_only: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Retrieve analytics data for a given date, presented as a compressed JSON file\n        https://api.slack.com/methods/admin.analytics.getFile\n        '
        kwargs.update({'type': type})
        if date is not None:
            kwargs.update({'date': date})
        if metadata_only is not None:
            kwargs.update({'metadata_only': metadata_only})
        return self.api_call('admin.analytics.getFile', params=kwargs)

    def admin_apps_approve(self, *, app_id: Optional[str]=None, request_id: Optional[str]=None, enterprise_id: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Approve an app for installation on a workspace.\n        Either app_id or request_id is required.\n        These IDs can be obtained either directly via the app_requested event,\n        or by the admin.apps.requests.list method.\n        https://api.slack.com/methods/admin.apps.approve\n        '
        if app_id:
            kwargs.update({'app_id': app_id})
        elif request_id:
            kwargs.update({'request_id': request_id})
        else:
            raise e.SlackRequestError('The app_id or request_id argument must be specified.')
        kwargs.update({'enterprise_id': enterprise_id, 'team_id': team_id})
        return self.api_call('admin.apps.approve', params=kwargs)

    def admin_apps_approved_list(self, *, cursor: Optional[str]=None, limit: Optional[int]=None, enterprise_id: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'List approved apps for an org or workspace.\n        https://api.slack.com/methods/admin.apps.approved.list\n        '
        kwargs.update({'cursor': cursor, 'limit': limit, 'enterprise_id': enterprise_id, 'team_id': team_id})
        return self.api_call('admin.apps.approved.list', http_verb='GET', params=kwargs)

    def admin_apps_clearResolution(self, *, app_id: str, enterprise_id: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Clear an app resolution\n        https://api.slack.com/methods/admin.apps.clearResolution\n        '
        kwargs.update({'app_id': app_id, 'enterprise_id': enterprise_id, 'team_id': team_id})
        return self.api_call('admin.apps.clearResolution', http_verb='POST', params=kwargs)

    def admin_apps_requests_cancel(self, *, request_id: str, enterprise_id: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'List app requests for a team/workspace.\n        https://api.slack.com/methods/admin.apps.requests.cancel\n        '
        kwargs.update({'request_id': request_id, 'enterprise_id': enterprise_id, 'team_id': team_id})
        return self.api_call('admin.apps.requests.cancel', http_verb='POST', params=kwargs)

    def admin_apps_requests_list(self, *, cursor: Optional[str]=None, limit: Optional[int]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'List app requests for a team/workspace.\n        https://api.slack.com/methods/admin.apps.requests.list\n        '
        kwargs.update({'cursor': cursor, 'limit': limit, 'team_id': team_id})
        return self.api_call('admin.apps.requests.list', http_verb='GET', params=kwargs)

    def admin_apps_restrict(self, *, app_id: Optional[str]=None, request_id: Optional[str]=None, enterprise_id: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Restrict an app for installation on a workspace.\n        Exactly one of the team_id or enterprise_id arguments is required, not both.\n        Either app_id or request_id is required. These IDs can be obtained either directly\n        via the app_requested event, or by the admin.apps.requests.list method.\n        https://api.slack.com/methods/admin.apps.restrict\n        '
        if app_id:
            kwargs.update({'app_id': app_id})
        elif request_id:
            kwargs.update({'request_id': request_id})
        else:
            raise e.SlackRequestError('The app_id or request_id argument must be specified.')
        kwargs.update({'enterprise_id': enterprise_id, 'team_id': team_id})
        return self.api_call('admin.apps.restrict', params=kwargs)

    def admin_apps_restricted_list(self, *, cursor: Optional[str]=None, limit: Optional[int]=None, enterprise_id: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'List restricted apps for an org or workspace.\n        https://api.slack.com/methods/admin.apps.restricted.list\n        '
        kwargs.update({'cursor': cursor, 'limit': limit, 'enterprise_id': enterprise_id, 'team_id': team_id})
        return self.api_call('admin.apps.restricted.list', http_verb='GET', params=kwargs)

    def admin_apps_uninstall(self, *, app_id: str, enterprise_id: Optional[str]=None, team_ids: Optional[Union[str, Sequence[str]]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Uninstall an app from one or many workspaces, or an entire enterprise organization.\n        With an org-level token, enterprise_id or team_ids is required.\n        https://api.slack.com/methods/admin.apps.uninstall\n        '
        kwargs.update({'app_id': app_id})
        if enterprise_id is not None:
            kwargs.update({'enterprise_id': enterprise_id})
        if team_ids is not None:
            if isinstance(team_ids, (list, Tuple)):
                kwargs.update({'team_ids': ','.join(team_ids)})
            else:
                kwargs.update({'team_ids': team_ids})
        return self.api_call('admin.apps.uninstall', http_verb='POST', params=kwargs)

    def admin_apps_activities_list(self, *, app_id: Optional[str]=None, component_id: Optional[str]=None, component_type: Optional[str]=None, log_event_type: Optional[str]=None, max_date_created: Optional[int]=None, min_date_created: Optional[int]=None, min_log_level: Optional[str]=None, sort_direction: Optional[str]=None, source: Optional[str]=None, team_id: Optional[str]=None, trace_id: Optional[str]=None, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Get logs for a specified team/org\n        https://api.slack.com/methods/admin.apps.activities.list\n        '
        kwargs.update({'app_id': app_id, 'component_id': component_id, 'component_type': component_type, 'log_event_type': log_event_type, 'max_date_created': max_date_created, 'min_date_created': min_date_created, 'min_log_level': min_log_level, 'sort_direction': sort_direction, 'source': source, 'team_id': team_id, 'trace_id': trace_id, 'cursor': cursor, 'limit': limit})
        return self.api_call('admin.apps.activities.list', params=kwargs)

    def admin_apps_config_lookup(self, *, app_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Look up the app config for connectors by their IDs\n        https://api.slack.com/methods/admin.apps.config.lookup\n        '
        if isinstance(app_ids, (list, Tuple)):
            kwargs.update({'app_ids': ','.join(app_ids)})
        else:
            kwargs.update({'app_ids': app_ids})
        return self.api_call('admin.apps.config.lookup', params=kwargs)

    def admin_apps_config_set(self, *, app_id: str, domain_restrictions: Optional[Dict[str, Any]]=None, workflow_auth_strategy: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Set the app config for a connector\n        https://api.slack.com/methods/admin.apps.config.set\n        '
        kwargs.update({'app_id': app_id, 'workflow_auth_strategy': workflow_auth_strategy})
        if domain_restrictions is not None:
            kwargs.update({'domain_restrictions': json.dumps(domain_restrictions)})
        return self.api_call('admin.apps.config.set', params=kwargs)

    def admin_auth_policy_getEntities(self, *, policy_name: str, cursor: Optional[str]=None, entity_type: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Fetch all the entities assigned to a particular authentication policy by name.\n        https://api.slack.com/methods/admin.auth.policy.getEntities\n        '
        kwargs.update({'policy_name': policy_name})
        if cursor is not None:
            kwargs.update({'cursor': cursor})
        if entity_type is not None:
            kwargs.update({'entity_type': entity_type})
        if limit is not None:
            kwargs.update({'limit': limit})
        return self.api_call('admin.auth.policy.getEntities', http_verb='POST', params=kwargs)

    def admin_auth_policy_assignEntities(self, *, entity_ids: Union[str, Sequence[str]], policy_name: str, entity_type: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Assign entities to a particular authentication policy.\n        https://api.slack.com/methods/admin.auth.policy.assignEntities\n        '
        if isinstance(entity_ids, (list, Tuple)):
            kwargs.update({'entity_ids': ','.join(entity_ids)})
        else:
            kwargs.update({'entity_ids': entity_ids})
        kwargs.update({'policy_name': policy_name})
        kwargs.update({'entity_type': entity_type})
        return self.api_call('admin.auth.policy.assignEntities', http_verb='POST', params=kwargs)

    def admin_auth_policy_removeEntities(self, *, entity_ids: Union[str, Sequence[str]], policy_name: str, entity_type: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Remove specified entities from a specified authentication policy.\n        https://api.slack.com/methods/admin.auth.policy.removeEntities\n        '
        if isinstance(entity_ids, (list, Tuple)):
            kwargs.update({'entity_ids': ','.join(entity_ids)})
        else:
            kwargs.update({'entity_ids': entity_ids})
        kwargs.update({'policy_name': policy_name})
        kwargs.update({'entity_type': entity_type})
        return self.api_call('admin.auth.policy.removeEntities', http_verb='POST', params=kwargs)

    def admin_barriers_create(self, *, barriered_from_usergroup_ids: Union[str, Sequence[str]], primary_usergroup_id: str, restricted_subjects: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Create an Information Barrier\n        https://api.slack.com/methods/admin.barriers.create\n        '
        kwargs.update({'primary_usergroup_id': primary_usergroup_id})
        if isinstance(barriered_from_usergroup_ids, (list, Tuple)):
            kwargs.update({'barriered_from_usergroup_ids': ','.join(barriered_from_usergroup_ids)})
        else:
            kwargs.update({'barriered_from_usergroup_ids': barriered_from_usergroup_ids})
        if isinstance(restricted_subjects, (list, Tuple)):
            kwargs.update({'restricted_subjects': ','.join(restricted_subjects)})
        else:
            kwargs.update({'restricted_subjects': restricted_subjects})
        return self.api_call('admin.barriers.create', http_verb='POST', params=kwargs)

    def admin_barriers_delete(self, *, barrier_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Delete an existing Information Barrier\n        https://api.slack.com/methods/admin.barriers.delete\n        '
        kwargs.update({'barrier_id': barrier_id})
        return self.api_call('admin.barriers.delete', http_verb='POST', params=kwargs)

    def admin_barriers_update(self, *, barrier_id: str, barriered_from_usergroup_ids: Union[str, Sequence[str]], primary_usergroup_id: str, restricted_subjects: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Update an existing Information Barrier\n        https://api.slack.com/methods/admin.barriers.update\n        '
        kwargs.update({'barrier_id': barrier_id, 'primary_usergroup_id': primary_usergroup_id})
        if isinstance(barriered_from_usergroup_ids, (list, Tuple)):
            kwargs.update({'barriered_from_usergroup_ids': ','.join(barriered_from_usergroup_ids)})
        else:
            kwargs.update({'barriered_from_usergroup_ids': barriered_from_usergroup_ids})
        if isinstance(restricted_subjects, (list, Tuple)):
            kwargs.update({'restricted_subjects': ','.join(restricted_subjects)})
        else:
            kwargs.update({'restricted_subjects': restricted_subjects})
        return self.api_call('admin.barriers.update', http_verb='POST', params=kwargs)

    def admin_barriers_list(self, *, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Get all Information Barriers for your organization\n        https://api.slack.com/methods/admin.barriers.list'
        kwargs.update({'cursor': cursor, 'limit': limit})
        return self.api_call('admin.barriers.list', http_verb='GET', params=kwargs)

    def admin_conversations_create(self, *, is_private: bool, name: str, description: Optional[str]=None, org_wide: Optional[bool]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Create a public or private channel-based conversation.\n        https://api.slack.com/methods/admin.conversations.create\n        '
        kwargs.update({'is_private': is_private, 'name': name, 'description': description, 'org_wide': org_wide, 'team_id': team_id})
        return self.api_call('admin.conversations.create', params=kwargs)

    def admin_conversations_delete(self, *, channel_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Delete a public or private channel.\n        https://api.slack.com/methods/admin.conversations.delete\n        '
        kwargs.update({'channel_id': channel_id})
        return self.api_call('admin.conversations.delete', params=kwargs)

    def admin_conversations_invite(self, *, channel_id: str, user_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Invite a user to a public or private channel.\n        https://api.slack.com/methods/admin.conversations.invite\n        '
        kwargs.update({'channel_id': channel_id})
        if isinstance(user_ids, (list, Tuple)):
            kwargs.update({'user_ids': ','.join(user_ids)})
        else:
            kwargs.update({'user_ids': user_ids})
        return self.api_call('admin.conversations.invite', params=kwargs)

    def admin_conversations_archive(self, *, channel_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Archive a public or private channel.\n        https://api.slack.com/methods/admin.conversations.archive\n        '
        kwargs.update({'channel_id': channel_id})
        return self.api_call('admin.conversations.archive', params=kwargs)

    def admin_conversations_unarchive(self, *, channel_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Unarchive a public or private channel.\n        https://api.slack.com/methods/admin.conversations.archive\n        '
        kwargs.update({'channel_id': channel_id})
        return self.api_call('admin.conversations.unarchive', params=kwargs)

    def admin_conversations_rename(self, *, channel_id: str, name: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Rename a public or private channel.\n        https://api.slack.com/methods/admin.conversations.rename\n        '
        kwargs.update({'channel_id': channel_id, 'name': name})
        return self.api_call('admin.conversations.rename', params=kwargs)

    def admin_conversations_search(self, *, cursor: Optional[str]=None, limit: Optional[int]=None, query: Optional[str]=None, search_channel_types: Optional[Union[str, Sequence[str]]]=None, sort: Optional[str]=None, sort_dir: Optional[str]=None, team_ids: Optional[Union[str, Sequence[str]]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Search for public or private channels in an Enterprise organization.\n        https://api.slack.com/methods/admin.conversations.search\n        '
        kwargs.update({'cursor': cursor, 'limit': limit, 'query': query, 'sort': sort, 'sort_dir': sort_dir})
        if isinstance(search_channel_types, (list, Tuple)):
            kwargs.update({'search_channel_types': ','.join(search_channel_types)})
        else:
            kwargs.update({'search_channel_types': search_channel_types})
        if isinstance(team_ids, (list, Tuple)):
            kwargs.update({'team_ids': ','.join(team_ids)})
        else:
            kwargs.update({'team_ids': team_ids})
        return self.api_call('admin.conversations.search', params=kwargs)

    def admin_conversations_convertToPrivate(self, *, channel_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Convert a public channel to a private channel.\n        https://api.slack.com/methods/admin.conversations.convertToPrivate\n        '
        kwargs.update({'channel_id': channel_id})
        return self.api_call('admin.conversations.convertToPrivate', params=kwargs)

    def admin_conversations_convertToPublic(self, *, channel_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Convert a privte channel to a public channel.\n        https://api.slack.com/methods/admin.conversations.convertToPublic\n        '
        kwargs.update({'channel_id': channel_id})
        return self.api_call('admin.conversations.convertToPublic', params=kwargs)

    def admin_conversations_setConversationPrefs(self, *, channel_id: str, prefs: Union[str, Dict[str, str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Set the posting permissions for a public or private channel.\n        https://api.slack.com/methods/admin.conversations.setConversationPrefs\n        '
        kwargs.update({'channel_id': channel_id})
        if isinstance(prefs, dict):
            kwargs.update({'prefs': json.dumps(prefs)})
        else:
            kwargs.update({'prefs': prefs})
        return self.api_call('admin.conversations.setConversationPrefs', params=kwargs)

    def admin_conversations_getConversationPrefs(self, *, channel_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Get conversation preferences for a public or private channel.\n        https://api.slack.com/methods/admin.conversations.getConversationPrefs\n        '
        kwargs.update({'channel_id': channel_id})
        return self.api_call('admin.conversations.getConversationPrefs', params=kwargs)

    def admin_conversations_disconnectShared(self, *, channel_id: str, leaving_team_ids: Optional[Union[str, Sequence[str]]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Disconnect a connected channel from one or more workspaces.\n        https://api.slack.com/methods/admin.conversations.disconnectShared\n        '
        kwargs.update({'channel_id': channel_id})
        if isinstance(leaving_team_ids, (list, Tuple)):
            kwargs.update({'leaving_team_ids': ','.join(leaving_team_ids)})
        else:
            kwargs.update({'leaving_team_ids': leaving_team_ids})
        return self.api_call('admin.conversations.disconnectShared', params=kwargs)

    def admin_conversations_lookup(self, *, last_message_activity_before: int, team_ids: Union[str, Sequence[str]], cursor: Optional[str]=None, limit: Optional[int]=None, max_member_count: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Returns channels on the given team using the filters.\n        https://api.slack.com/methods/admin.conversations.lookup\n        '
        kwargs.update({'last_message_activity_before': last_message_activity_before, 'cursor': cursor, 'limit': limit, 'max_member_count': max_member_count})
        if isinstance(team_ids, (list, Tuple)):
            kwargs.update({'team_ids': ','.join(team_ids)})
        else:
            kwargs.update({'team_ids': team_ids})
        return self.api_call('admin.conversations.lookup', params=kwargs)

    def admin_conversations_ekm_listOriginalConnectedChannelInfo(self, *, channel_ids: Optional[Union[str, Sequence[str]]]=None, cursor: Optional[str]=None, limit: Optional[int]=None, team_ids: Optional[Union[str, Sequence[str]]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'List all disconnected channels—i.e.,\n        channels that were once connected to other workspaces and then disconnected—and\n        the corresponding original channel IDs for key revocation with EKM.\n        https://api.slack.com/methods/admin.conversations.ekm.listOriginalConnectedChannelInfo\n        '
        kwargs.update({'cursor': cursor, 'limit': limit})
        if isinstance(channel_ids, (list, Tuple)):
            kwargs.update({'channel_ids': ','.join(channel_ids)})
        else:
            kwargs.update({'channel_ids': channel_ids})
        if isinstance(team_ids, (list, Tuple)):
            kwargs.update({'team_ids': ','.join(team_ids)})
        else:
            kwargs.update({'team_ids': team_ids})
        return self.api_call('admin.conversations.ekm.listOriginalConnectedChannelInfo', params=kwargs)

    def admin_conversations_restrictAccess_addGroup(self, *, channel_id: str, group_id: str, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Add an allowlist of IDP groups for accessing a channel.\n        https://api.slack.com/methods/admin.conversations.restrictAccess.addGroup\n        '
        kwargs.update({'channel_id': channel_id, 'group_id': group_id, 'team_id': team_id})
        return self.api_call('admin.conversations.restrictAccess.addGroup', http_verb='GET', params=kwargs)

    def admin_conversations_restrictAccess_listGroups(self, *, channel_id: str, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'List all IDP Groups linked to a channel.\n        https://api.slack.com/methods/admin.conversations.restrictAccess.listGroups\n        '
        kwargs.update({'channel_id': channel_id, 'team_id': team_id})
        return self.api_call('admin.conversations.restrictAccess.listGroups', http_verb='GET', params=kwargs)

    def admin_conversations_restrictAccess_removeGroup(self, *, channel_id: str, group_id: str, team_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Remove a linked IDP group linked from a private channel.\n        https://api.slack.com/methods/admin.conversations.restrictAccess.removeGroup\n        '
        kwargs.update({'channel_id': channel_id, 'group_id': group_id, 'team_id': team_id})
        return self.api_call('admin.conversations.restrictAccess.removeGroup', http_verb='GET', params=kwargs)

    def admin_conversations_setTeams(self, *, channel_id: str, org_channel: Optional[bool]=None, target_team_ids: Optional[Union[str, Sequence[str]]]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Set the workspaces in an Enterprise grid org that connect to a public or private channel.\n        https://api.slack.com/methods/admin.conversations.setTeams\n        '
        kwargs.update({'channel_id': channel_id, 'org_channel': org_channel, 'team_id': team_id})
        if isinstance(target_team_ids, (list, Tuple)):
            kwargs.update({'target_team_ids': ','.join(target_team_ids)})
        else:
            kwargs.update({'target_team_ids': target_team_ids})
        return self.api_call('admin.conversations.setTeams', params=kwargs)

    def admin_conversations_getTeams(self, *, channel_id: str, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Set the workspaces in an Enterprise grid org that connect to a channel.\n        https://api.slack.com/methods/admin.conversations.getTeams\n        '
        kwargs.update({'channel_id': channel_id, 'cursor': cursor, 'limit': limit})
        return self.api_call('admin.conversations.getTeams', params=kwargs)

    def admin_conversations_getCustomRetention(self, *, channel_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        "Get a channel's retention policy\n        https://api.slack.com/methods/admin.conversations.getCustomRetention\n        "
        kwargs.update({'channel_id': channel_id})
        return self.api_call('admin.conversations.getCustomRetention', params=kwargs)

    def admin_conversations_removeCustomRetention(self, *, channel_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        "Remove a channel's retention policy\n        https://api.slack.com/methods/admin.conversations.removeCustomRetention\n        "
        kwargs.update({'channel_id': channel_id})
        return self.api_call('admin.conversations.removeCustomRetention', params=kwargs)

    def admin_conversations_setCustomRetention(self, *, channel_id: str, duration_days: int, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        "Set a channel's retention policy\n        https://api.slack.com/methods/admin.conversations.setCustomRetention\n        "
        kwargs.update({'channel_id': channel_id, 'duration_days': duration_days})
        return self.api_call('admin.conversations.setCustomRetention', params=kwargs)

    def admin_conversations_bulkArchive(self, *, channel_ids: Union[Sequence[str], str], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Archive public or private channels in bulk.\n        https://api.slack.com/methods/admin.conversations.bulkArchive\n        '
        kwargs.update({'channel_ids': ','.join(channel_ids) if isinstance(channel_ids, (list, tuple)) else channel_ids})
        return self.api_call('admin.conversations.bulkArchive', params=kwargs)

    def admin_conversations_bulkDelete(self, *, channel_ids: Union[Sequence[str], str], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Delete public or private channels in bulk.\n        https://slack.com/api/admin.conversations.bulkDelete\n        '
        kwargs.update({'channel_ids': ','.join(channel_ids) if isinstance(channel_ids, (list, tuple)) else channel_ids})
        return self.api_call('admin.conversations.bulkDelete', params=kwargs)

    def admin_conversations_bulkMove(self, *, channel_ids: Union[Sequence[str], str], target_team_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Move public or private channels in bulk.\n        https://api.slack.com/methods/admin.conversations.bulkMove\n        '
        kwargs.update({'target_team_id': target_team_id, 'channel_ids': ','.join(channel_ids) if isinstance(channel_ids, (list, tuple)) else channel_ids})
        return self.api_call('admin.conversations.bulkMove', params=kwargs)

    def admin_emoji_add(self, *, name: str, url: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Add an emoji.\n        https://api.slack.com/methods/admin.emoji.add\n        '
        kwargs.update({'name': name, 'url': url})
        return self.api_call('admin.emoji.add', http_verb='GET', params=kwargs)

    def admin_emoji_addAlias(self, *, alias_for: str, name: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Add an emoji alias.\n        https://api.slack.com/methods/admin.emoji.addAlias\n        '
        kwargs.update({'alias_for': alias_for, 'name': name})
        return self.api_call('admin.emoji.addAlias', http_verb='GET', params=kwargs)

    def admin_emoji_list(self, *, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'List emoji for an Enterprise Grid organization.\n        https://api.slack.com/methods/admin.emoji.list\n        '
        kwargs.update({'cursor': cursor, 'limit': limit})
        return self.api_call('admin.emoji.list', http_verb='GET', params=kwargs)

    def admin_emoji_remove(self, *, name: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Remove an emoji across an Enterprise Grid organization.\n        https://api.slack.com/methods/admin.emoji.remove\n        '
        kwargs.update({'name': name})
        return self.api_call('admin.emoji.remove', http_verb='GET', params=kwargs)

    def admin_emoji_rename(self, *, name: str, new_name: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Rename an emoji.\n        https://api.slack.com/methods/admin.emoji.rename\n        '
        kwargs.update({'name': name, 'new_name': new_name})
        return self.api_call('admin.emoji.rename', http_verb='GET', params=kwargs)

    def admin_functions_list(self, *, app_ids: Union[str, Sequence[str]], team_id: Optional[str]=None, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Look up functions by a set of apps\n        https://api.slack.com/methods/admin.functions.list\n        '
        if isinstance(app_ids, (list, Tuple)):
            kwargs.update({'app_ids': ','.join(app_ids)})
        else:
            kwargs.update({'app_ids': app_ids})
        kwargs.update({'team_id': team_id, 'cursor': cursor, 'limit': limit})
        return self.api_call('admin.functions.list', params=kwargs)

    def admin_functions_permissions_lookup(self, *, function_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Lookup the visibility of multiple Slack functions\n        and include the users if it is limited to particular named entities.\n        https://api.slack.com/methods/admin.functions.permissions.lookup\n        '
        if isinstance(function_ids, (list, Tuple)):
            kwargs.update({'function_ids': ','.join(function_ids)})
        else:
            kwargs.update({'function_ids': function_ids})
        return self.api_call('admin.functions.permissions.lookup', params=kwargs)

    def admin_functions_permissions_set(self, *, function_id: str, visibility: str, user_ids: Optional[Union[str, Sequence[str]]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Set the visibility of a Slack function\n        and define the users or workspaces if it is set to named_entities\n        https://api.slack.com/methods/admin.functions.permissions.set\n        '
        kwargs.update({'function_id': function_id, 'visibility': visibility})
        if user_ids is not None:
            if isinstance(user_ids, (list, Tuple)):
                kwargs.update({'user_ids': ','.join(user_ids)})
            else:
                kwargs.update({'user_ids': user_ids})
        return self.api_call('admin.functions.permissions.set', params=kwargs)

    def admin_roles_addAssignments(self, *, role_id: str, entity_ids: Union[str, Sequence[str]], user_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Adds members to the specified role with the specified scopes\n        https://api.slack.com/methods/admin.roles.addAssignments\n        '
        kwargs.update({'role_id': role_id})
        if isinstance(entity_ids, (list, Tuple)):
            kwargs.update({'entity_ids': ','.join(entity_ids)})
        else:
            kwargs.update({'entity_ids': entity_ids})
        if isinstance(user_ids, (list, Tuple)):
            kwargs.update({'user_ids': ','.join(user_ids)})
        else:
            kwargs.update({'user_ids': user_ids})
        return self.api_call('admin.roles.addAssignments', params=kwargs)

    def admin_roles_listAssignments(self, *, role_ids: Optional[Union[str, Sequence[str]]]=None, entity_ids: Optional[Union[str, Sequence[str]]]=None, cursor: Optional[str]=None, limit: Optional[Union[str, int]]=None, sort_dir: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Lists assignments for all roles across entities.\n            Options to scope results by any combination of roles or entities\n        https://api.slack.com/methods/admin.roles.listAssignments\n        '
        kwargs.update({'cursor': cursor, 'limit': limit, 'sort_dir': sort_dir})
        if isinstance(entity_ids, (list, Tuple)):
            kwargs.update({'entity_ids': ','.join(entity_ids)})
        else:
            kwargs.update({'entity_ids': entity_ids})
        if isinstance(role_ids, (list, Tuple)):
            kwargs.update({'role_ids': ','.join(role_ids)})
        else:
            kwargs.update({'role_ids': role_ids})
        return self.api_call('admin.roles.listAssignments', params=kwargs)

    def admin_roles_removeAssignments(self, *, role_id: str, entity_ids: Union[str, Sequence[str]], user_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Removes a set of users from a role for the given scopes and entities\n        https://api.slack.com/methods/admin.roles.removeAssignments\n        '
        kwargs.update({'role_id': role_id})
        if isinstance(entity_ids, (list, Tuple)):
            kwargs.update({'entity_ids': ','.join(entity_ids)})
        else:
            kwargs.update({'entity_ids': entity_ids})
        if isinstance(user_ids, (list, Tuple)):
            kwargs.update({'user_ids': ','.join(user_ids)})
        else:
            kwargs.update({'user_ids': user_ids})
        return self.api_call('admin.roles.removeAssignments', params=kwargs)

    def admin_users_session_reset(self, *, user_id: str, mobile_only: Optional[bool]=None, web_only: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Wipes all valid sessions on all devices for a given user.\n        https://api.slack.com/methods/admin.users.session.reset\n        '
        kwargs.update({'user_id': user_id, 'mobile_only': mobile_only, 'web_only': web_only})
        return self.api_call('admin.users.session.reset', params=kwargs)

    def admin_users_session_resetBulk(self, *, user_ids: Union[str, Sequence[str]], mobile_only: Optional[bool]=None, web_only: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Enqueues an asynchronous job to wipe all valid sessions on all devices for a given list of users\n        https://api.slack.com/methods/admin.users.session.resetBulk\n        '
        if isinstance(user_ids, (list, Tuple)):
            kwargs.update({'user_ids': ','.join(user_ids)})
        else:
            kwargs.update({'user_ids': user_ids})
        kwargs.update({'mobile_only': mobile_only, 'web_only': web_only})
        return self.api_call('admin.users.session.resetBulk', params=kwargs)

    def admin_users_session_invalidate(self, *, session_id: str, team_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Invalidate a single session for a user by session_id.\n        https://api.slack.com/methods/admin.users.session.invalidate\n        '
        kwargs.update({'session_id': session_id, 'team_id': team_id})
        return self.api_call('admin.users.session.invalidate', params=kwargs)

    def admin_users_session_list(self, *, cursor: Optional[str]=None, limit: Optional[int]=None, team_id: Optional[str]=None, user_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Lists all active user sessions for an organization\n        https://api.slack.com/methods/admin.users.session.list\n        '
        kwargs.update({'cursor': cursor, 'limit': limit, 'team_id': team_id, 'user_id': user_id})
        return self.api_call('admin.users.session.list', params=kwargs)

    def admin_teams_settings_setDefaultChannels(self, *, team_id: str, channel_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Set the default channels of a workspace.\n        https://api.slack.com/methods/admin.teams.settings.setDefaultChannels\n        '
        kwargs.update({'team_id': team_id})
        if isinstance(channel_ids, (list, Tuple)):
            kwargs.update({'channel_ids': ','.join(channel_ids)})
        else:
            kwargs.update({'channel_ids': channel_ids})
        return self.api_call('admin.teams.settings.setDefaultChannels', http_verb='GET', params=kwargs)

    def admin_users_session_getSettings(self, *, user_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Get user-specific session settings—the session duration\n        and what happens when the client closes—given a list of users.\n        https://api.slack.com/methods/admin.users.session.getSettings\n        '
        if isinstance(user_ids, (list, Tuple)):
            kwargs.update({'user_ids': ','.join(user_ids)})
        else:
            kwargs.update({'user_ids': user_ids})
        return self.api_call('admin.users.session.getSettings', params=kwargs)

    def admin_users_session_setSettings(self, *, user_ids: Union[str, Sequence[str]], desktop_app_browser_quit: Optional[bool]=None, duration: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Configure the user-level session settings—the session duration\n        and what happens when the client closes—for one or more users.\n        https://api.slack.com/methods/admin.users.session.setSettings\n        '
        if isinstance(user_ids, (list, Tuple)):
            kwargs.update({'user_ids': ','.join(user_ids)})
        else:
            kwargs.update({'user_ids': user_ids})
        kwargs.update({'desktop_app_browser_quit': desktop_app_browser_quit, 'duration': duration})
        return self.api_call('admin.users.session.setSettings', params=kwargs)

    def admin_users_session_clearSettings(self, *, user_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Clear user-specific session settings—the session duration\n        and what happens when the client closes—for a list of users.\n        https://api.slack.com/methods/admin.users.session.clearSettings\n        '
        if isinstance(user_ids, (list, Tuple)):
            kwargs.update({'user_ids': ','.join(user_ids)})
        else:
            kwargs.update({'user_ids': user_ids})
        return self.api_call('admin.users.session.clearSettings', params=kwargs)

    def admin_users_unsupportedVersions_export(self, *, date_end_of_support: Optional[Union[str, int]]=None, date_sessions_started: Optional[Union[str, int]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Ask Slackbot to send you an export listing all workspace members using unsupported software,\n        presented as a zipped CSV file.\n        https://api.slack.com/methods/admin.users.unsupportedVersions.export\n        '
        kwargs.update({'date_end_of_support': date_end_of_support, 'date_sessions_started': date_sessions_started})
        return self.api_call('admin.users.unsupportedVersions.export', params=kwargs)

    def admin_inviteRequests_approve(self, *, invite_request_id: str, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Approve a workspace invite request.\n        https://api.slack.com/methods/admin.inviteRequests.approve\n        '
        kwargs.update({'invite_request_id': invite_request_id, 'team_id': team_id})
        return self.api_call('admin.inviteRequests.approve', params=kwargs)

    def admin_inviteRequests_approved_list(self, *, cursor: Optional[str]=None, limit: Optional[int]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'List all approved workspace invite requests.\n        https://api.slack.com/methods/admin.inviteRequests.approved.list\n        '
        kwargs.update({'cursor': cursor, 'limit': limit, 'team_id': team_id})
        return self.api_call('admin.inviteRequests.approved.list', params=kwargs)

    def admin_inviteRequests_denied_list(self, *, cursor: Optional[str]=None, limit: Optional[int]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'List all denied workspace invite requests.\n        https://api.slack.com/methods/admin.inviteRequests.denied.list\n        '
        kwargs.update({'cursor': cursor, 'limit': limit, 'team_id': team_id})
        return self.api_call('admin.inviteRequests.denied.list', params=kwargs)

    def admin_inviteRequests_deny(self, *, invite_request_id: str, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Deny a workspace invite request.\n        https://api.slack.com/methods/admin.inviteRequests.deny\n        '
        kwargs.update({'invite_request_id': invite_request_id, 'team_id': team_id})
        return self.api_call('admin.inviteRequests.deny', params=kwargs)

    def admin_inviteRequests_list(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'List all pending workspace invite requests.'
        return self.api_call('admin.inviteRequests.list', params=kwargs)

    def admin_teams_admins_list(self, *, team_id: str, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'List all of the admins on a given workspace.\n        https://api.slack.com/methods/admin.inviteRequests.list\n        '
        kwargs.update({'cursor': cursor, 'limit': limit, 'team_id': team_id})
        return self.api_call('admin.teams.admins.list', http_verb='GET', params=kwargs)

    def admin_teams_create(self, *, team_domain: str, team_name: str, team_description: Optional[str]=None, team_discoverability: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Create an Enterprise team.\n        https://api.slack.com/methods/admin.teams.create\n        '
        kwargs.update({'team_domain': team_domain, 'team_name': team_name, 'team_description': team_description, 'team_discoverability': team_discoverability})
        return self.api_call('admin.teams.create', params=kwargs)

    def admin_teams_list(self, *, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'List all teams on an Enterprise organization.\n        https://api.slack.com/methods/admin.teams.list\n        '
        kwargs.update({'cursor': cursor, 'limit': limit})
        return self.api_call('admin.teams.list', params=kwargs)

    def admin_teams_owners_list(self, *, team_id: str, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'List all of the admins on a given workspace.\n        https://api.slack.com/methods/admin.teams.owners.list\n        '
        kwargs.update({'team_id': team_id, 'cursor': cursor, 'limit': limit})
        return self.api_call('admin.teams.owners.list', http_verb='GET', params=kwargs)

    def admin_teams_settings_info(self, *, team_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Fetch information about settings in a workspace\n        https://api.slack.com/methods/admin.teams.settings.info\n        '
        kwargs.update({'team_id': team_id})
        return self.api_call('admin.teams.settings.info', params=kwargs)

    def admin_teams_settings_setDescription(self, *, team_id: str, description: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Set the description of a given workspace.\n        https://api.slack.com/methods/admin.teams.settings.setDescription\n        '
        kwargs.update({'team_id': team_id, 'description': description})
        return self.api_call('admin.teams.settings.setDescription', params=kwargs)

    def admin_teams_settings_setDiscoverability(self, *, team_id: str, discoverability: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Sets the icon of a workspace.\n        https://api.slack.com/methods/admin.teams.settings.setDiscoverability\n        '
        kwargs.update({'team_id': team_id, 'discoverability': discoverability})
        return self.api_call('admin.teams.settings.setDiscoverability', params=kwargs)

    def admin_teams_settings_setIcon(self, *, team_id: str, image_url: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Sets the icon of a workspace.\n        https://api.slack.com/methods/admin.teams.settings.setIcon\n        '
        kwargs.update({'team_id': team_id, 'image_url': image_url})
        return self.api_call('admin.teams.settings.setIcon', http_verb='GET', params=kwargs)

    def admin_teams_settings_setName(self, *, team_id: str, name: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Sets the icon of a workspace.\n        https://api.slack.com/methods/admin.teams.settings.setName\n        '
        kwargs.update({'team_id': team_id, 'name': name})
        return self.api_call('admin.teams.settings.setName', params=kwargs)

    def admin_usergroups_addChannels(self, *, channel_ids: Union[str, Sequence[str]], usergroup_id: str, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Add one or more default channels to an IDP group.\n        https://api.slack.com/methods/admin.usergroups.addChannels\n        '
        kwargs.update({'team_id': team_id, 'usergroup_id': usergroup_id})
        if isinstance(channel_ids, (list, Tuple)):
            kwargs.update({'channel_ids': ','.join(channel_ids)})
        else:
            kwargs.update({'channel_ids': channel_ids})
        return self.api_call('admin.usergroups.addChannels', params=kwargs)

    def admin_usergroups_addTeams(self, *, usergroup_id: str, team_ids: Union[str, Sequence[str]], auto_provision: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Associate one or more default workspaces with an organization-wide IDP group.\n        https://api.slack.com/methods/admin.usergroups.addTeams\n        '
        kwargs.update({'usergroup_id': usergroup_id, 'auto_provision': auto_provision})
        if isinstance(team_ids, (list, Tuple)):
            kwargs.update({'team_ids': ','.join(team_ids)})
        else:
            kwargs.update({'team_ids': team_ids})
        return self.api_call('admin.usergroups.addTeams', params=kwargs)

    def admin_usergroups_listChannels(self, *, usergroup_id: str, include_num_members: Optional[bool]=None, team_id: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Add one or more default channels to an IDP group.\n        https://api.slack.com/methods/admin.usergroups.listChannels\n        '
        kwargs.update({'usergroup_id': usergroup_id, 'include_num_members': include_num_members, 'team_id': team_id})
        return self.api_call('admin.usergroups.listChannels', params=kwargs)

    def admin_usergroups_removeChannels(self, *, usergroup_id: str, channel_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Add one or more default channels to an IDP group.\n        https://api.slack.com/methods/admin.usergroups.removeChannels\n        '
        kwargs.update({'usergroup_id': usergroup_id})
        if isinstance(channel_ids, (list, Tuple)):
            kwargs.update({'channel_ids': ','.join(channel_ids)})
        else:
            kwargs.update({'channel_ids': channel_ids})
        return self.api_call('admin.usergroups.removeChannels', params=kwargs)

    def admin_users_assign(self, *, team_id: str, user_id: str, channel_ids: Optional[Union[str, Sequence[str]]]=None, is_restricted: Optional[bool]=None, is_ultra_restricted: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Add an Enterprise user to a workspace.\n        https://api.slack.com/methods/admin.users.assign\n        '
        kwargs.update({'team_id': team_id, 'user_id': user_id, 'is_restricted': is_restricted, 'is_ultra_restricted': is_ultra_restricted})
        if isinstance(channel_ids, (list, Tuple)):
            kwargs.update({'channel_ids': ','.join(channel_ids)})
        else:
            kwargs.update({'channel_ids': channel_ids})
        return self.api_call('admin.users.assign', params=kwargs)

    def admin_users_invite(self, *, team_id: str, email: str, channel_ids: Union[str, Sequence[str]], custom_message: Optional[str]=None, email_password_policy_enabled: Optional[bool]=None, guest_expiration_ts: Optional[Union[str, float]]=None, is_restricted: Optional[bool]=None, is_ultra_restricted: Optional[bool]=None, real_name: Optional[str]=None, resend: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Invite a user to a workspace.\n        https://api.slack.com/methods/admin.users.invite\n        '
        kwargs.update({'team_id': team_id, 'email': email, 'custom_message': custom_message, 'email_password_policy_enabled': email_password_policy_enabled, 'guest_expiration_ts': str(guest_expiration_ts) if guest_expiration_ts is not None else None, 'is_restricted': is_restricted, 'is_ultra_restricted': is_ultra_restricted, 'real_name': real_name, 'resend': resend})
        if isinstance(channel_ids, (list, Tuple)):
            kwargs.update({'channel_ids': ','.join(channel_ids)})
        else:
            kwargs.update({'channel_ids': channel_ids})
        return self.api_call('admin.users.invite', params=kwargs)

    def admin_users_list(self, *, team_id: str, include_deactivated_user_workspaces: Optional[bool]=None, is_active: Optional[bool]=None, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'List users on a workspace\n        https://api.slack.com/methods/admin.users.list\n        '
        kwargs.update({'team_id': team_id, 'include_deactivated_user_workspaces': include_deactivated_user_workspaces, 'is_active': is_active, 'cursor': cursor, 'limit': limit})
        return self.api_call('admin.users.list', params=kwargs)

    def admin_users_remove(self, *, team_id: str, user_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Remove a user from a workspace.\n        https://api.slack.com/methods/admin.users.remove\n        '
        kwargs.update({'team_id': team_id, 'user_id': user_id})
        return self.api_call('admin.users.remove', params=kwargs)

    def admin_users_setAdmin(self, *, team_id: str, user_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Set an existing guest, regular user, or owner to be an admin user.\n        https://api.slack.com/methods/admin.users.setAdmin\n        '
        kwargs.update({'team_id': team_id, 'user_id': user_id})
        return self.api_call('admin.users.setAdmin', params=kwargs)

    def admin_users_setExpiration(self, *, expiration_ts: int, user_id: str, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Set an expiration for a guest user.\n        https://api.slack.com/methods/admin.users.setExpiration\n        '
        kwargs.update({'expiration_ts': expiration_ts, 'team_id': team_id, 'user_id': user_id})
        return self.api_call('admin.users.setExpiration', params=kwargs)

    def admin_users_setOwner(self, *, team_id: str, user_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Set an existing guest, regular user, or admin user to be a workspace owner.\n        https://api.slack.com/methods/admin.users.setOwner\n        '
        kwargs.update({'team_id': team_id, 'user_id': user_id})
        return self.api_call('admin.users.setOwner', params=kwargs)

    def admin_users_setRegular(self, *, team_id: str, user_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Set an existing guest user, admin user, or owner to be a regular user.\n        https://api.slack.com/methods/admin.users.setRegular\n        '
        kwargs.update({'team_id': team_id, 'user_id': user_id})
        return self.api_call('admin.users.setRegular', params=kwargs)

    def admin_workflows_search(self, *, app_id: Optional[str]=None, collaborator_ids: Optional[Union[str, Sequence[str]]]=None, cursor: Optional[str]=None, limit: Optional[int]=None, no_collaborators: Optional[bool]=None, num_trigger_ids: Optional[int]=None, query: Optional[str]=None, sort: Optional[str]=None, sort_dir: Optional[str]=None, source: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Search workflows within the team or enterprise\n        https://api.slack.com/methods/admin.workflows.search\n        '
        if collaborator_ids is not None:
            if isinstance(collaborator_ids, (list, Tuple)):
                kwargs.update({'collaborator_ids': ','.join(collaborator_ids)})
            else:
                kwargs.update({'collaborator_ids': collaborator_ids})
        kwargs.update({'app_id': app_id, 'cursor': cursor, 'limit': limit, 'no_collaborators': no_collaborators, 'num_trigger_ids': num_trigger_ids, 'query': query, 'sort': sort, 'sort_dir': sort_dir, 'source': source})
        return self.api_call('admin.workflows.search', params=kwargs)

    def admin_workflows_permissions_lookup(self, *, workflow_ids: Union[str, Sequence[str]], max_workflow_triggers: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Look up the permissions for a set of workflows\n        https://api.slack.com/methods/admin.workflows.permissions.lookup\n        '
        if isinstance(workflow_ids, (list, Tuple)):
            kwargs.update({'workflow_ids': ','.join(workflow_ids)})
        else:
            kwargs.update({'workflow_ids': workflow_ids})
        kwargs.update({'max_workflow_triggers': max_workflow_triggers})
        return self.api_call('admin.workflows.permissions.lookup', params=kwargs)

    def admin_workflows_collaborators_add(self, *, collaborator_ids: Union[str, Sequence[str]], workflow_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Add collaborators to workflows within the team or enterprise\n        https://api.slack.com/methods/admin.workflows.collaborators.add\n        '
        if isinstance(collaborator_ids, (list, Tuple)):
            kwargs.update({'collaborator_ids': ','.join(collaborator_ids)})
        else:
            kwargs.update({'collaborator_ids': collaborator_ids})
        if isinstance(workflow_ids, (list, Tuple)):
            kwargs.update({'workflow_ids': ','.join(workflow_ids)})
        else:
            kwargs.update({'workflow_ids': workflow_ids})
        return self.api_call('admin.workflows.collaborators.add', params=kwargs)

    def admin_workflows_collaborators_remove(self, *, collaborator_ids: Union[str, Sequence[str]], workflow_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Remove collaborators from workflows within the team or enterprise\n        https://api.slack.com/methods/admin.workflows.collaborators.remove\n        '
        if isinstance(collaborator_ids, (list, Tuple)):
            kwargs.update({'collaborator_ids': ','.join(collaborator_ids)})
        else:
            kwargs.update({'collaborator_ids': collaborator_ids})
        if isinstance(workflow_ids, (list, Tuple)):
            kwargs.update({'workflow_ids': ','.join(workflow_ids)})
        else:
            kwargs.update({'workflow_ids': workflow_ids})
        return self.api_call('admin.workflows.collaborators.remove', params=kwargs)

    def admin_workflows_unpublish(self, *, workflow_ids: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Unpublish workflows within the team or enterprise\n        https://api.slack.com/methods/admin.workflows.unpublish\n        '
        if isinstance(workflow_ids, (list, Tuple)):
            kwargs.update({'workflow_ids': ','.join(workflow_ids)})
        else:
            kwargs.update({'workflow_ids': workflow_ids})
        return self.api_call('admin.workflows.unpublish', params=kwargs)

    def api_test(self, *, error: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Checks API calling code.\n        https://api.slack.com/methods/api.test\n        '
        kwargs.update({'error': error})
        return self.api_call('api.test', params=kwargs)

    def apps_connections_open(self, *, app_token: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Generate a temporary Socket Mode WebSocket URL that your app can connect to\n        in order to receive events and interactive payloads\n        https://api.slack.com/methods/apps.connections.open\n        '
        kwargs.update({'token': app_token})
        return self.api_call('apps.connections.open', http_verb='POST', params=kwargs)

    def apps_event_authorizations_list(self, *, event_context: str, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Get a list of authorizations for the given event context.\n        Each authorization represents an app installation that the event is visible to.\n        https://api.slack.com/methods/apps.event.authorizations.list\n        '
        kwargs.update({'event_context': event_context, 'cursor': cursor, 'limit': limit})
        return self.api_call('apps.event.authorizations.list', params=kwargs)

    def apps_uninstall(self, *, client_id: str, client_secret: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Uninstalls your app from a workspace.\n        https://api.slack.com/methods/apps.uninstall\n        '
        kwargs.update({'client_id': client_id, 'client_secret': client_secret})
        return self.api_call('apps.uninstall', params=kwargs)

    def auth_revoke(self, *, test: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Revokes a token.\n        https://api.slack.com/methods/auth.revoke\n        '
        kwargs.update({'test': test})
        return self.api_call('auth.revoke', http_verb='GET', params=kwargs)

    def auth_test(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Checks authentication & identity.\n        https://api.slack.com/methods/auth.test\n        '
        return self.api_call('auth.test', params=kwargs)

    def auth_teams_list(self, cursor: Optional[str]=None, limit: Optional[int]=None, include_icon: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'List the workspaces a token can access.\n        https://api.slack.com/methods/auth.teams.list\n        '
        kwargs.update({'cursor': cursor, 'limit': limit, 'include_icon': include_icon})
        return self.api_call('auth.teams.list', params=kwargs)

    def bookmarks_add(self, *, channel_id: str, title: str, type: str, emoji: Optional[str]=None, entity_id: Optional[str]=None, link: Optional[str]=None, parent_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Add bookmark to a channel.\n        https://api.slack.com/methods/bookmarks.add\n        '
        kwargs.update({'channel_id': channel_id, 'title': title, 'type': type, 'emoji': emoji, 'entity_id': entity_id, 'link': link, 'parent_id': parent_id})
        return self.api_call('bookmarks.add', http_verb='POST', params=kwargs)

    def bookmarks_edit(self, *, bookmark_id: str, channel_id: str, emoji: Optional[str]=None, link: Optional[str]=None, title: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Edit bookmark.\n        https://api.slack.com/methods/bookmarks.edit\n        '
        kwargs.update({'bookmark_id': bookmark_id, 'channel_id': channel_id, 'emoji': emoji, 'link': link, 'title': title})
        return self.api_call('bookmarks.edit', http_verb='POST', params=kwargs)

    def bookmarks_list(self, *, channel_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'List bookmark for the channel.\n        https://api.slack.com/methods/bookmarks.list\n        '
        kwargs.update({'channel_id': channel_id})
        return self.api_call('bookmarks.list', http_verb='POST', params=kwargs)

    def bookmarks_remove(self, *, bookmark_id: str, channel_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Remove bookmark from the channel.\n        https://api.slack.com/methods/bookmarks.remove\n        '
        kwargs.update({'bookmark_id': bookmark_id, 'channel_id': channel_id})
        return self.api_call('bookmarks.remove', http_verb='POST', params=kwargs)

    def bots_info(self, *, bot: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Gets information about a bot user.\n        https://api.slack.com/methods/bots.info\n        '
        kwargs.update({'bot': bot, 'team_id': team_id})
        return self.api_call('bots.info', http_verb='GET', params=kwargs)

    def calls_add(self, *, external_unique_id: str, join_url: str, created_by: Optional[str]=None, date_start: Optional[int]=None, desktop_app_join_url: Optional[str]=None, external_display_id: Optional[str]=None, title: Optional[str]=None, users: Optional[Union[str, Sequence[Dict[str, str]]]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Registers a new Call.\n        https://api.slack.com/methods/calls.add\n        '
        kwargs.update({'external_unique_id': external_unique_id, 'join_url': join_url, 'created_by': created_by, 'date_start': date_start, 'desktop_app_join_url': desktop_app_join_url, 'external_display_id': external_display_id, 'title': title})
        _update_call_participants(kwargs, users if users is not None else kwargs.get('users'))
        return self.api_call('calls.add', http_verb='POST', params=kwargs)

    def calls_end(self, *, id: str, duration: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Ends a Call.\n        https://api.slack.com/methods/calls.end\n        '
        kwargs.update({'id': id, 'duration': duration})
        return self.api_call('calls.end', http_verb='POST', params=kwargs)

    def calls_info(self, *, id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Returns information about a Call.\n        https://api.slack.com/methods/calls.info\n        '
        kwargs.update({'id': id})
        return self.api_call('calls.info', http_verb='POST', params=kwargs)

    def calls_participants_add(self, *, id: str, users: Union[str, Sequence[Dict[str, str]]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Registers new participants added to a Call.\n        https://api.slack.com/methods/calls.participants.add\n        '
        kwargs.update({'id': id})
        _update_call_participants(kwargs, users)
        return self.api_call('calls.participants.add', http_verb='POST', params=kwargs)

    def calls_participants_remove(self, *, id: str, users: Union[str, Sequence[Dict[str, str]]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Registers participants removed from a Call.\n        https://api.slack.com/methods/calls.participants.remove\n        '
        kwargs.update({'id': id})
        _update_call_participants(kwargs, users)
        return self.api_call('calls.participants.remove', http_verb='POST', params=kwargs)

    def calls_update(self, *, id: str, desktop_app_join_url: Optional[str]=None, join_url: Optional[str]=None, title: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Updates information about a Call.\n        https://api.slack.com/methods/calls.update\n        '
        kwargs.update({'id': id, 'desktop_app_join_url': desktop_app_join_url, 'join_url': join_url, 'title': title})
        return self.api_call('calls.update', http_verb='POST', params=kwargs)

    def channels_archive(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Archives a channel.'
        kwargs.update({'channel': channel})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('channels.archive', json=kwargs)

    def channels_create(self, *, name: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Creates a channel.'
        kwargs.update({'name': name})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('channels.create', json=kwargs)

    def channels_history(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Fetches history of messages and events from a channel.'
        kwargs.update({'channel': channel})
        return self.api_call('channels.history', http_verb='GET', params=kwargs)

    def channels_info(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Gets information about a channel.'
        kwargs.update({'channel': channel})
        return self.api_call('channels.info', http_verb='GET', params=kwargs)

    def channels_invite(self, *, channel: str, user: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Invites a user to a channel.'
        kwargs.update({'channel': channel, 'user': user})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('channels.invite', json=kwargs)

    def channels_join(self, *, name: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Joins a channel, creating it if needed.'
        kwargs.update({'name': name})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('channels.join', json=kwargs)

    def channels_kick(self, *, channel: str, user: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Removes a user from a channel.'
        kwargs.update({'channel': channel, 'user': user})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('channels.kick', json=kwargs)

    def channels_leave(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Leaves a channel.'
        kwargs.update({'channel': channel})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('channels.leave', json=kwargs)

    def channels_list(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Lists all channels in a Slack team.'
        return self.api_call('channels.list', http_verb='GET', params=kwargs)

    def channels_mark(self, *, channel: str, ts: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Sets the read cursor in a channel.'
        kwargs.update({'channel': channel, 'ts': ts})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('channels.mark', json=kwargs)

    def channels_rename(self, *, channel: str, name: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Renames a channel.'
        kwargs.update({'channel': channel, 'name': name})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('channels.rename', json=kwargs)

    def channels_replies(self, *, channel: str, thread_ts: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Retrieve a thread of messages posted to a channel'
        kwargs.update({'channel': channel, 'thread_ts': thread_ts})
        return self.api_call('channels.replies', http_verb='GET', params=kwargs)

    def channels_setPurpose(self, *, channel: str, purpose: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Sets the purpose for a channel.'
        kwargs.update({'channel': channel, 'purpose': purpose})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('channels.setPurpose', json=kwargs)

    def channels_setTopic(self, *, channel: str, topic: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Sets the topic for a channel.'
        kwargs.update({'channel': channel, 'topic': topic})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('channels.setTopic', json=kwargs)

    def channels_unarchive(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Unarchives a channel.'
        kwargs.update({'channel': channel})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('channels.unarchive', json=kwargs)

    def chat_delete(self, *, channel: str, ts: str, as_user: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Deletes a message.\n        https://api.slack.com/methods/chat.delete\n        '
        kwargs.update({'channel': channel, 'ts': ts, 'as_user': as_user})
        return self.api_call('chat.delete', params=kwargs)

    def chat_deleteScheduledMessage(self, *, channel: str, scheduled_message_id: str, as_user: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Deletes a scheduled message.\n        https://api.slack.com/methods/chat.deleteScheduledMessage\n        '
        kwargs.update({'channel': channel, 'scheduled_message_id': scheduled_message_id, 'as_user': as_user})
        return self.api_call('chat.deleteScheduledMessage', params=kwargs)

    def chat_getPermalink(self, *, channel: str, message_ts: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Retrieve a permalink URL for a specific extant message\n        https://api.slack.com/methods/chat.getPermalink\n        '
        kwargs.update({'channel': channel, 'message_ts': message_ts})
        return self.api_call('chat.getPermalink', http_verb='GET', params=kwargs)

    def chat_meMessage(self, *, channel: str, text: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Share a me message into a channel.\n        https://api.slack.com/methods/chat.meMessage\n        '
        kwargs.update({'channel': channel, 'text': text})
        return self.api_call('chat.meMessage', params=kwargs)

    def chat_postEphemeral(self, *, channel: str, user: str, text: Optional[str]=None, as_user: Optional[bool]=None, attachments: Optional[Union[str, Sequence[Union[Dict, Attachment]]]]=None, blocks: Optional[Union[str, Sequence[Union[Dict, Block]]]]=None, thread_ts: Optional[str]=None, icon_emoji: Optional[str]=None, icon_url: Optional[str]=None, link_names: Optional[bool]=None, username: Optional[str]=None, parse: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Sends an ephemeral message to a user in a channel.\n        https://api.slack.com/methods/chat.postEphemeral\n        '
        kwargs.update({'channel': channel, 'user': user, 'text': text, 'as_user': as_user, 'attachments': attachments, 'blocks': blocks, 'thread_ts': thread_ts, 'icon_emoji': icon_emoji, 'icon_url': icon_url, 'link_names': link_names, 'username': username, 'parse': parse})
        _parse_web_class_objects(kwargs)
        kwargs = _remove_none_values(kwargs)
        _warn_if_text_or_attachment_fallback_is_missing('chat.postEphemeral', kwargs)
        return self.api_call('chat.postEphemeral', json=kwargs)

    def chat_postMessage(self, *, channel: str, text: Optional[str]=None, as_user: Optional[bool]=None, attachments: Optional[Union[str, Sequence[Union[Dict, Attachment]]]]=None, blocks: Optional[Union[str, Sequence[Union[Dict, Block]]]]=None, thread_ts: Optional[str]=None, reply_broadcast: Optional[bool]=None, unfurl_links: Optional[bool]=None, unfurl_media: Optional[bool]=None, container_id: Optional[str]=None, icon_emoji: Optional[str]=None, icon_url: Optional[str]=None, mrkdwn: Optional[bool]=None, link_names: Optional[bool]=None, username: Optional[str]=None, parse: Optional[str]=None, metadata: Optional[Union[Dict, Metadata]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Sends a message to a channel.\n        https://api.slack.com/methods/chat.postMessage\n        '
        kwargs.update({'channel': channel, 'text': text, 'as_user': as_user, 'attachments': attachments, 'blocks': blocks, 'thread_ts': thread_ts, 'reply_broadcast': reply_broadcast, 'unfurl_links': unfurl_links, 'unfurl_media': unfurl_media, 'container_id': container_id, 'icon_emoji': icon_emoji, 'icon_url': icon_url, 'mrkdwn': mrkdwn, 'link_names': link_names, 'username': username, 'parse': parse, 'metadata': metadata})
        _parse_web_class_objects(kwargs)
        kwargs = _remove_none_values(kwargs)
        _warn_if_text_or_attachment_fallback_is_missing('chat.postMessage', kwargs)
        return self.api_call('chat.postMessage', json=kwargs)

    def chat_scheduleMessage(self, *, channel: str, post_at: Union[str, int], text: str, as_user: Optional[bool]=None, attachments: Optional[Union[str, Sequence[Union[Dict, Attachment]]]]=None, blocks: Optional[Union[str, Sequence[Union[Dict, Block]]]]=None, thread_ts: Optional[str]=None, parse: Optional[str]=None, reply_broadcast: Optional[bool]=None, unfurl_links: Optional[bool]=None, unfurl_media: Optional[bool]=None, link_names: Optional[bool]=None, metadata: Optional[Union[Dict, Metadata]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Schedules a message.\n        https://api.slack.com/methods/chat.scheduleMessage\n        '
        kwargs.update({'channel': channel, 'post_at': post_at, 'text': text, 'as_user': as_user, 'attachments': attachments, 'blocks': blocks, 'thread_ts': thread_ts, 'reply_broadcast': reply_broadcast, 'parse': parse, 'unfurl_links': unfurl_links, 'unfurl_media': unfurl_media, 'link_names': link_names, 'metadata': metadata})
        _parse_web_class_objects(kwargs)
        kwargs = _remove_none_values(kwargs)
        _warn_if_text_or_attachment_fallback_is_missing('chat.scheduleMessage', kwargs)
        return self.api_call('chat.scheduleMessage', json=kwargs)

    def chat_unfurl(self, *, channel: Optional[str]=None, ts: Optional[str]=None, source: Optional[str]=None, unfurl_id: Optional[str]=None, unfurls: Optional[Dict[str, Dict]]=None, user_auth_blocks: Optional[Union[str, Sequence[Union[Dict, Block]]]]=None, user_auth_message: Optional[str]=None, user_auth_required: Optional[bool]=None, user_auth_url: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Provide custom unfurl behavior for user-posted URLs.\n        https://api.slack.com/methods/chat.unfurl\n        '
        kwargs.update({'channel': channel, 'ts': ts, 'source': source, 'unfurl_id': unfurl_id, 'unfurls': unfurls, 'user_auth_blocks': user_auth_blocks, 'user_auth_message': user_auth_message, 'user_auth_required': user_auth_required, 'user_auth_url': user_auth_url})
        _parse_web_class_objects(kwargs)
        kwargs = _remove_none_values(kwargs)
        return self.api_call('chat.unfurl', json=kwargs)

    def chat_update(self, *, channel: str, ts: str, text: Optional[str]=None, attachments: Optional[Union[str, Sequence[Union[Dict, Attachment]]]]=None, blocks: Optional[Union[str, Sequence[Union[Dict, Block]]]]=None, as_user: Optional[bool]=None, file_ids: Optional[Union[str, Sequence[str]]]=None, link_names: Optional[bool]=None, parse: Optional[str]=None, reply_broadcast: Optional[bool]=None, metadata: Optional[Union[Dict, Metadata]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Updates a message in a channel.\n        https://api.slack.com/methods/chat.update\n        '
        kwargs.update({'channel': channel, 'ts': ts, 'text': text, 'attachments': attachments, 'blocks': blocks, 'as_user': as_user, 'link_names': link_names, 'parse': parse, 'reply_broadcast': reply_broadcast, 'metadata': metadata})
        if isinstance(file_ids, (list, Tuple)):
            kwargs.update({'file_ids': ','.join(file_ids)})
        else:
            kwargs.update({'file_ids': file_ids})
        _parse_web_class_objects(kwargs)
        kwargs = _remove_none_values(kwargs)
        _warn_if_text_or_attachment_fallback_is_missing('chat.update', kwargs)
        return self.api_call('chat.update', json=kwargs)

    def chat_scheduledMessages_list(self, *, channel: Optional[str]=None, cursor: Optional[str]=None, latest: Optional[str]=None, limit: Optional[int]=None, oldest: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Lists all scheduled messages.\n        https://api.slack.com/methods/chat.scheduledMessages.list\n        '
        kwargs.update({'channel': channel, 'cursor': cursor, 'latest': latest, 'limit': limit, 'oldest': oldest, 'team_id': team_id})
        return self.api_call('chat.scheduledMessages.list', params=kwargs)

    def conversations_acceptSharedInvite(self, *, channel_name: str, channel_id: Optional[str]=None, invite_id: Optional[str]=None, free_trial_accepted: Optional[bool]=None, is_private: Optional[bool]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Accepts an invitation to a Slack Connect channel.\n        https://api.slack.com/methods/conversations.acceptSharedInvite\n        '
        if channel_id is None and invite_id is None:
            raise e.SlackRequestError('Either channel_id or invite_id must be provided.')
        kwargs.update({'channel_name': channel_name, 'channel_id': channel_id, 'invite_id': invite_id, 'free_trial_accepted': free_trial_accepted, 'is_private': is_private, 'team_id': team_id})
        return self.api_call('conversations.acceptSharedInvite', http_verb='POST', params=kwargs)

    def conversations_approveSharedInvite(self, *, invite_id: str, target_team: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Approves an invitation to a Slack Connect channel.\n        https://api.slack.com/methods/conversations.approveSharedInvite\n        '
        kwargs.update({'invite_id': invite_id, 'target_team': target_team})
        return self.api_call('conversations.approveSharedInvite', http_verb='POST', params=kwargs)

    def conversations_archive(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Archives a conversation.\n        https://api.slack.com/methods/conversations.archive\n        '
        kwargs.update({'channel': channel})
        return self.api_call('conversations.archive', params=kwargs)

    def conversations_close(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Closes a direct message or multi-person direct message.\n        https://api.slack.com/methods/conversations.close\n        '
        kwargs.update({'channel': channel})
        return self.api_call('conversations.close', params=kwargs)

    def conversations_create(self, *, name: str, is_private: Optional[bool]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Initiates a public or private channel-based conversation\n        https://api.slack.com/methods/conversations.create\n        '
        kwargs.update({'name': name, 'is_private': is_private, 'team_id': team_id})
        return self.api_call('conversations.create', params=kwargs)

    def conversations_declineSharedInvite(self, *, invite_id: str, target_team: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Declines a Slack Connect channel invite.\n        https://api.slack.com/methods/conversations.declineSharedInvite\n        '
        kwargs.update({'invite_id': invite_id, 'target_team': target_team})
        return self.api_call('conversations.declineSharedInvite', http_verb='GET', params=kwargs)

    def conversations_history(self, *, channel: str, cursor: Optional[str]=None, inclusive: Optional[bool]=None, include_all_metadata: Optional[bool]=None, latest: Optional[str]=None, limit: Optional[int]=None, oldest: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        "Fetches a conversation's history of messages and events.\n        https://api.slack.com/methods/conversations.history\n        "
        kwargs.update({'channel': channel, 'cursor': cursor, 'inclusive': inclusive, 'include_all_metadata': include_all_metadata, 'limit': limit, 'latest': latest, 'oldest': oldest})
        return self.api_call('conversations.history', http_verb='GET', params=kwargs)

    def conversations_info(self, *, channel: str, include_locale: Optional[bool]=None, include_num_members: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Retrieve information about a conversation.\n        https://api.slack.com/methods/conversations.info\n        '
        kwargs.update({'channel': channel, 'include_locale': include_locale, 'include_num_members': include_num_members})
        return self.api_call('conversations.info', http_verb='GET', params=kwargs)

    def conversations_invite(self, *, channel: str, users: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Invites users to a channel.\n        https://api.slack.com/methods/conversations.invite\n        '
        kwargs.update({'channel': channel})
        if isinstance(users, (list, Tuple)):
            kwargs.update({'users': ','.join(users)})
        else:
            kwargs.update({'users': users})
        return self.api_call('conversations.invite', params=kwargs)

    def conversations_inviteShared(self, *, channel: str, emails: Optional[Union[str, Sequence[str]]]=None, user_ids: Optional[Union[str, Sequence[str]]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Sends an invitation to a Slack Connect channel.\n        https://api.slack.com/methods/conversations.inviteShared\n        '
        if emails is None and user_ids is None:
            raise e.SlackRequestError('Either emails or user ids must be provided.')
        kwargs.update({'channel': channel})
        if isinstance(emails, (list, Tuple)):
            kwargs.update({'emails': ','.join(emails)})
        else:
            kwargs.update({'emails': emails})
        if isinstance(user_ids, (list, Tuple)):
            kwargs.update({'user_ids': ','.join(user_ids)})
        else:
            kwargs.update({'user_ids': user_ids})
        return self.api_call('conversations.inviteShared', http_verb='GET', params=kwargs)

    def conversations_join(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Joins an existing conversation.\n        https://api.slack.com/methods/conversations.join\n        '
        kwargs.update({'channel': channel})
        return self.api_call('conversations.join', params=kwargs)

    def conversations_kick(self, *, channel: str, user: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Removes a user from a conversation.\n        https://api.slack.com/methods/conversations.kick\n        '
        kwargs.update({'channel': channel, 'user': user})
        return self.api_call('conversations.kick', params=kwargs)

    def conversations_leave(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Leaves a conversation.\n        https://api.slack.com/methods/conversations.leave\n        '
        kwargs.update({'channel': channel})
        return self.api_call('conversations.leave', params=kwargs)

    def conversations_list(self, *, cursor: Optional[str]=None, exclude_archived: Optional[bool]=None, limit: Optional[int]=None, team_id: Optional[str]=None, types: Optional[Union[str, Sequence[str]]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Lists all channels in a Slack team.\n        https://api.slack.com/methods/conversations.list\n        '
        kwargs.update({'cursor': cursor, 'exclude_archived': exclude_archived, 'limit': limit, 'team_id': team_id})
        if isinstance(types, (list, Tuple)):
            kwargs.update({'types': ','.join(types)})
        else:
            kwargs.update({'types': types})
        return self.api_call('conversations.list', http_verb='GET', params=kwargs)

    def conversations_listConnectInvites(self, *, count: Optional[int]=None, cursor: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'List shared channel invites that have been generated\n        or received but have not yet been approved by all parties.\n        https://api.slack.com/methods/conversations.listConnectInvites\n        '
        kwargs.update({'count': count, 'cursor': cursor, 'team_id': team_id})
        return self.api_call('conversations.listConnectInvites', params=kwargs)

    def conversations_mark(self, *, channel: str, ts: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Sets the read cursor in a channel.\n        https://api.slack.com/methods/conversations.mark\n        '
        kwargs.update({'channel': channel, 'ts': ts})
        return self.api_call('conversations.mark', params=kwargs)

    def conversations_members(self, *, channel: str, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Retrieve members of a conversation.\n        https://api.slack.com/methods/conversations.members\n        '
        kwargs.update({'channel': channel, 'cursor': cursor, 'limit': limit})
        return self.api_call('conversations.members', http_verb='GET', params=kwargs)

    def conversations_open(self, *, channel: Optional[str]=None, return_im: Optional[bool]=None, users: Optional[Union[str, Sequence[str]]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Opens or resumes a direct message or multi-person direct message.\n        https://api.slack.com/methods/conversations.open\n        '
        if channel is None and users is None:
            raise e.SlackRequestError('Either channel or users must be provided.')
        kwargs.update({'channel': channel, 'return_im': return_im})
        if isinstance(users, (list, Tuple)):
            kwargs.update({'users': ','.join(users)})
        else:
            kwargs.update({'users': users})
        return self.api_call('conversations.open', params=kwargs)

    def conversations_rename(self, *, channel: str, name: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Renames a conversation.\n        https://api.slack.com/methods/conversations.rename\n        '
        kwargs.update({'channel': channel, 'name': name})
        return self.api_call('conversations.rename', params=kwargs)

    def conversations_replies(self, *, channel: str, ts: str, cursor: Optional[str]=None, inclusive: Optional[bool]=None, include_all_metadata: Optional[bool]=None, latest: Optional[str]=None, limit: Optional[int]=None, oldest: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Retrieve a thread of messages posted to a conversation\n        https://api.slack.com/methods/conversations.replies\n        '
        kwargs.update({'channel': channel, 'ts': ts, 'cursor': cursor, 'inclusive': inclusive, 'include_all_metadata': include_all_metadata, 'limit': limit, 'latest': latest, 'oldest': oldest})
        return self.api_call('conversations.replies', http_verb='GET', params=kwargs)

    def conversations_setPurpose(self, *, channel: str, purpose: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Sets the purpose for a conversation.\n        https://api.slack.com/methods/conversations.setPurpose\n        '
        kwargs.update({'channel': channel, 'purpose': purpose})
        return self.api_call('conversations.setPurpose', params=kwargs)

    def conversations_setTopic(self, *, channel: str, topic: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Sets the topic for a conversation.\n        https://api.slack.com/methods/conversations.setTopic\n        '
        kwargs.update({'channel': channel, 'topic': topic})
        return self.api_call('conversations.setTopic', params=kwargs)

    def conversations_unarchive(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Reverses conversation archival.\n        https://api.slack.com/methods/conversations.unarchive\n        '
        kwargs.update({'channel': channel})
        return self.api_call('conversations.unarchive', params=kwargs)

    def dialog_open(self, *, dialog: Dict[str, Any], trigger_id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Open a dialog with a user.\n        https://api.slack.com/methods/dialog.open\n        '
        kwargs.update({'dialog': dialog, 'trigger_id': trigger_id})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('dialog.open', json=kwargs)

    def dnd_endDnd(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        "Ends the current user's Do Not Disturb session immediately.\n        https://api.slack.com/methods/dnd.endDnd\n        "
        return self.api_call('dnd.endDnd', params=kwargs)

    def dnd_endSnooze(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        "Ends the current user's snooze mode immediately.\n        https://api.slack.com/methods/dnd.endSnooze\n        "
        return self.api_call('dnd.endSnooze', params=kwargs)

    def dnd_info(self, *, team_id: Optional[str]=None, user: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        "Retrieves a user's current Do Not Disturb status.\n        https://api.slack.com/methods/dnd.info\n        "
        kwargs.update({'team_id': team_id, 'user': user})
        return self.api_call('dnd.info', http_verb='GET', params=kwargs)

    def dnd_setSnooze(self, *, num_minutes: Union[int, str], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Turns on Do Not Disturb mode for the current user, or changes its duration.\n        https://api.slack.com/methods/dnd.setSnooze\n        '
        kwargs.update({'num_minutes': num_minutes})
        return self.api_call('dnd.setSnooze', http_verb='GET', params=kwargs)

    def dnd_teamInfo(self, users: Union[str, Sequence[str]], team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Retrieves the Do Not Disturb status for users on a team.\n        https://api.slack.com/methods/dnd.teamInfo\n        '
        if isinstance(users, (list, Tuple)):
            kwargs.update({'users': ','.join(users)})
        else:
            kwargs.update({'users': users})
        kwargs.update({'team_id': team_id})
        return self.api_call('dnd.teamInfo', http_verb='GET', params=kwargs)

    def emoji_list(self, include_categories: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Lists custom emoji for a team.\n        https://api.slack.com/methods/emoji.list\n        '
        kwargs.update({'include_categories': include_categories})
        return self.api_call('emoji.list', http_verb='GET', params=kwargs)

    def files_comments_delete(self, *, file: str, id: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Deletes an existing comment on a file.\n        https://api.slack.com/methods/files.comments.delete\n        '
        kwargs.update({'file': file, 'id': id})
        return self.api_call('files.comments.delete', params=kwargs)

    def files_delete(self, *, file: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Deletes a file.\n        https://api.slack.com/methods/files.delete\n        '
        kwargs.update({'file': file})
        return self.api_call('files.delete', params=kwargs)

    def files_info(self, *, file: str, count: Optional[int]=None, cursor: Optional[str]=None, limit: Optional[int]=None, page: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Gets information about a team file.\n        https://api.slack.com/methods/files.info\n        '
        kwargs.update({'file': file, 'count': count, 'cursor': cursor, 'limit': limit, 'page': page})
        return self.api_call('files.info', http_verb='GET', params=kwargs)

    def files_list(self, *, channel: Optional[str]=None, count: Optional[int]=None, page: Optional[int]=None, show_files_hidden_by_limit: Optional[bool]=None, team_id: Optional[str]=None, ts_from: Optional[str]=None, ts_to: Optional[str]=None, types: Optional[Union[str, Sequence[str]]]=None, user: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Lists & filters team files.\n        https://api.slack.com/methods/files.list\n        '
        kwargs.update({'channel': channel, 'count': count, 'page': page, 'show_files_hidden_by_limit': show_files_hidden_by_limit, 'team_id': team_id, 'ts_from': ts_from, 'ts_to': ts_to, 'user': user})
        if isinstance(types, (list, Tuple)):
            kwargs.update({'types': ','.join(types)})
        else:
            kwargs.update({'types': types})
        return self.api_call('files.list', http_verb='GET', params=kwargs)

    def files_remote_info(self, *, external_id: Optional[str]=None, file: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Retrieve information about a remote file added to Slack.\n        https://api.slack.com/methods/files.remote.info\n        '
        kwargs.update({'external_id': external_id, 'file': file})
        return self.api_call('files.remote.info', http_verb='GET', params=kwargs)

    def files_remote_list(self, *, channel: Optional[str]=None, cursor: Optional[str]=None, limit: Optional[int]=None, ts_from: Optional[str]=None, ts_to: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Retrieve information about a remote file added to Slack.\n        https://api.slack.com/methods/files.remote.list\n        '
        kwargs.update({'channel': channel, 'cursor': cursor, 'limit': limit, 'ts_from': ts_from, 'ts_to': ts_to})
        return self.api_call('files.remote.list', http_verb='GET', params=kwargs)

    def files_remote_add(self, *, external_id: str, external_url: str, title: str, filetype: Optional[str]=None, indexable_file_contents: Optional[Union[str, bytes, IOBase]]=None, preview_image: Optional[Union[str, bytes, IOBase]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Adds a file from a remote service.\n        https://api.slack.com/methods/files.remote.add\n        '
        kwargs.update({'external_id': external_id, 'external_url': external_url, 'title': title, 'filetype': filetype})
        files = None
        if preview_image is not None or indexable_file_contents is not None:
            files = {'preview_image': preview_image, 'indexable_file_contents': indexable_file_contents}
        return self.api_call('files.remote.add', http_verb='POST', data=kwargs, files=files)

    def files_remote_update(self, *, external_id: Optional[str]=None, external_url: Optional[str]=None, file: Optional[str]=None, title: Optional[str]=None, filetype: Optional[str]=None, indexable_file_contents: Optional[str]=None, preview_image: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Updates an existing remote file.\n        https://api.slack.com/methods/files.remote.update\n        '
        kwargs.update({'external_id': external_id, 'external_url': external_url, 'file': file, 'title': title, 'filetype': filetype})
        files = None
        if preview_image is not None or indexable_file_contents is not None:
            files = {'preview_image': preview_image, 'indexable_file_contents': indexable_file_contents}
        return self.api_call('files.remote.update', http_verb='POST', data=kwargs, files=files)

    def files_remote_remove(self, *, external_id: Optional[str]=None, file: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Remove a remote file.\n        https://api.slack.com/methods/files.remote.remove\n        '
        kwargs.update({'external_id': external_id, 'file': file})
        return self.api_call('files.remote.remove', http_verb='POST', params=kwargs)

    def files_remote_share(self, *, channels: Union[str, Sequence[str]], external_id: Optional[str]=None, file: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Share a remote file into a channel.\n        https://api.slack.com/methods/files.remote.share\n        '
        if external_id is None and file is None:
            raise e.SlackRequestError('Either external_id or file must be provided.')
        if isinstance(channels, (list, Tuple)):
            kwargs.update({'channels': ','.join(channels)})
        else:
            kwargs.update({'channels': channels})
        kwargs.update({'external_id': external_id, 'file': file})
        return self.api_call('files.remote.share', http_verb='GET', params=kwargs)

    def files_revokePublicURL(self, *, file: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Revokes public/external sharing access for a file\n        https://api.slack.com/methods/files.revokePublicURL\n        '
        kwargs.update({'file': file})
        return self.api_call('files.revokePublicURL', params=kwargs)

    def files_sharedPublicURL(self, *, file: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Enables a file for public/external sharing.\n        https://api.slack.com/methods/files.sharedPublicURL\n        '
        kwargs.update({'file': file})
        return self.api_call('files.sharedPublicURL', params=kwargs)

    def files_upload(self, *, file: Optional[Union[str, bytes, IOBase]]=None, content: Optional[Union[str, bytes]]=None, filename: Optional[str]=None, filetype: Optional[str]=None, initial_comment: Optional[str]=None, thread_ts: Optional[str]=None, title: Optional[str]=None, channels: Optional[Union[str, Sequence[str]]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Uploads or creates a file.\n        https://api.slack.com/methods/files.upload\n        '
        _print_files_upload_v2_suggestion()
        if file is None and content is None:
            raise e.SlackRequestError('The file or content argument must be specified.')
        if file is not None and content is not None:
            raise e.SlackRequestError('You cannot specify both the file and the content argument.')
        if isinstance(channels, (list, Tuple)):
            kwargs.update({'channels': ','.join(channels)})
        else:
            kwargs.update({'channels': channels})
        kwargs.update({'filename': filename, 'filetype': filetype, 'initial_comment': initial_comment, 'thread_ts': thread_ts, 'title': title})
        if file:
            if kwargs.get('filename') is None and isinstance(file, str):
                if kwargs.get('filename') is None:
                    kwargs['filename'] = file.split(os.path.sep)[-1]
            return self.api_call('files.upload', files={'file': file}, data=kwargs)
        else:
            kwargs['content'] = content
            return self.api_call('files.upload', data=kwargs)

    def files_upload_v2(self, *, filename: Optional[str]=None, file: Optional[Union[str, bytes, IOBase]]=None, content: Optional[Union[str, bytes]]=None, title: Optional[str]=None, alt_txt: Optional[str]=None, snippet_type: Optional[str]=None, file_uploads: Optional[List[Dict[str, Any]]]=None, channel: Optional[str]=None, initial_comment: Optional[str]=None, thread_ts: Optional[str]=None, request_file_info: bool=True, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'This wrapper method provides an easy way to upload files using the following endpoints:\n\n        - step1: https://api.slack.com/methods/files.getUploadURLExternal\n\n        - step2: "https://files.slack.com/upload/v1/..." URLs returned from files.getUploadURLExternal API\n\n        - step3: https://api.slack.com/methods/files.completeUploadExternal\n            and https://api.slack.com/methods/files.info\n\n        '
        if file is None and content is None and (file_uploads is None):
            raise e.SlackRequestError('Any of file, content, and file_uploads must be specified.')
        if file is not None and content is not None:
            raise e.SlackRequestError('You cannot specify both the file and the content argument.')
        (channels, filetype) = (kwargs.get('channels'), kwargs.get('filetype'))
        if channels is not None:
            warnings.warn('Although the channels parameter is still supported for smooth migration from legacy files.upload, we recommend using the new channel parameter with a single str value instead for more clarity.')
            if isinstance(channels, (list, Tuple)) and len(channels) > 1 or (isinstance(channels, str) and len(channels.split(',')) > 1):
                raise e.SlackRequestError('Sharing files with multiple channels is no longer supported in v2. Share files in each channel separately instead.')
        if filetype is not None:
            warnings.warn('The filetype parameter is no longer supported. Please remove it from the arguments.')
        files: List[Dict[str, Any]] = []
        if file_uploads is not None:
            for f in file_uploads:
                files.append(_to_v2_file_upload_item(f))
        else:
            f = _to_v2_file_upload_item({'filename': filename, 'file': file, 'content': content, 'title': title, 'alt_txt': alt_txt, 'snippet_type': snippet_type})
            files.append(f)
        for f in files:
            url_response = self.files_getUploadURLExternal(filename=f.get('filename'), length=f.get('length'), alt_txt=f.get('alt_txt'), snippet_type=f.get('snippet_type'), token=kwargs.get('token'))
            _validate_for_legacy_client(url_response)
            f['file_id'] = url_response.get('file_id')
            f['upload_url'] = url_response.get('upload_url')
        for f in files:
            upload_result = _upload_file_via_v2_url(url=f['upload_url'], data=f['data'], logger=self._logger, timeout=self.timeout, proxy=self.proxy, ssl=self.ssl)
            if upload_result.get('status') != 200:
                status = upload_result.get('status')
                body = upload_result.get('body')
                message = f"Failed to upload a file (status: {status}, body: {body}, filename: {f.get('filename')}, title: {f.get('title')})"
                raise e.SlackRequestError(message)
        channel_to_share = channel
        if channels is not None:
            if isinstance(channels, str):
                channel_to_share = channels.split(',')[0]
            else:
                channel_to_share = channels[0]
        completion = self.files_completeUploadExternal(files=[{'id': f['file_id'], 'title': f['title']} for f in files], channel_id=channel_to_share, initial_comment=initial_comment, thread_ts=thread_ts, **kwargs)
        if len(completion.get('files')) == 1:
            completion.data['file'] = completion.get('files')[0]
        return completion

    def files_getUploadURLExternal(self, *, filename: str, length: int, alt_txt: Optional[str]=None, snippet_type: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Gets a URL for an edge external upload.\n        https://api.slack.com/methods/files.getUploadURLExternal\n        '
        kwargs.update({'filename': filename, 'length': length, 'alt_txt': alt_txt, 'snippet_type': snippet_type})
        return self.api_call('files.getUploadURLExternal', params=kwargs)

    def files_completeUploadExternal(self, *, files: List[Dict[str, str]], channel_id: Optional[str]=None, initial_comment: Optional[str]=None, thread_ts: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Finishes an upload started with files.getUploadURLExternal.\n        https://api.slack.com/methods/files.completeUploadExternal\n        '
        _files = [{k: v for (k, v) in f.items() if v is not None} for f in files]
        kwargs.update({'files': json.dumps(_files), 'channel_id': channel_id, 'initial_comment': initial_comment, 'thread_ts': thread_ts})
        return self.api_call('files.completeUploadExternal', params=kwargs)

    def groups_archive(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Archives a private channel.'
        kwargs.update({'channel': channel})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('groups.archive', json=kwargs)

    def groups_create(self, *, name: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Creates a private channel.'
        kwargs.update({'name': name})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('groups.create', json=kwargs)

    def groups_createChild(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Clones and archives a private channel.'
        kwargs.update({'channel': channel})
        return self.api_call('groups.createChild', http_verb='GET', params=kwargs)

    def groups_history(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Fetches history of messages and events from a private channel.'
        kwargs.update({'channel': channel})
        return self.api_call('groups.history', http_verb='GET', params=kwargs)

    def groups_info(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Gets information about a private channel.'
        kwargs.update({'channel': channel})
        return self.api_call('groups.info', http_verb='GET', params=kwargs)

    def groups_invite(self, *, channel: str, user: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Invites a user to a private channel.'
        kwargs.update({'channel': channel, 'user': user})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('groups.invite', json=kwargs)

    def groups_kick(self, *, channel: str, user: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Removes a user from a private channel.'
        kwargs.update({'channel': channel, 'user': user})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('groups.kick', json=kwargs)

    def groups_leave(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Leaves a private channel.'
        kwargs.update({'channel': channel})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('groups.leave', json=kwargs)

    def groups_list(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Lists private channels that the calling user has access to.'
        return self.api_call('groups.list', http_verb='GET', params=kwargs)

    def groups_mark(self, *, channel: str, ts: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Sets the read cursor in a private channel.'
        kwargs.update({'channel': channel, 'ts': ts})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('groups.mark', json=kwargs)

    def groups_open(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Opens a private channel.'
        kwargs.update({'channel': channel})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('groups.open', json=kwargs)

    def groups_rename(self, *, channel: str, name: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Renames a private channel.'
        kwargs.update({'channel': channel, 'name': name})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('groups.rename', json=kwargs)

    def groups_replies(self, *, channel: str, thread_ts: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Retrieve a thread of messages posted to a private channel'
        kwargs.update({'channel': channel, 'thread_ts': thread_ts})
        return self.api_call('groups.replies', http_verb='GET', params=kwargs)

    def groups_setPurpose(self, *, channel: str, purpose: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Sets the purpose for a private channel.'
        kwargs.update({'channel': channel, 'purpose': purpose})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('groups.setPurpose', json=kwargs)

    def groups_setTopic(self, *, channel: str, topic: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Sets the topic for a private channel.'
        kwargs.update({'channel': channel, 'topic': topic})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('groups.setTopic', json=kwargs)

    def groups_unarchive(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Unarchives a private channel.'
        kwargs.update({'channel': channel})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('groups.unarchive', json=kwargs)

    def im_close(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Close a direct message channel.'
        kwargs.update({'channel': channel})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('im.close', json=kwargs)

    def im_history(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Fetches history of messages and events from direct message channel.'
        kwargs.update({'channel': channel})
        return self.api_call('im.history', http_verb='GET', params=kwargs)

    def im_list(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Lists direct message channels for the calling user.'
        return self.api_call('im.list', http_verb='GET', params=kwargs)

    def im_mark(self, *, channel: str, ts: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Sets the read cursor in a direct message channel.'
        kwargs.update({'channel': channel, 'ts': ts})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('im.mark', json=kwargs)

    def im_open(self, *, user: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Opens a direct message channel.'
        kwargs.update({'user': user})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('im.open', json=kwargs)

    def im_replies(self, *, channel: str, thread_ts: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Retrieve a thread of messages posted to a direct message conversation'
        kwargs.update({'channel': channel, 'thread_ts': thread_ts})
        return self.api_call('im.replies', http_verb='GET', params=kwargs)

    def migration_exchange(self, *, users: Union[str, Sequence[str]], team_id: Optional[str]=None, to_old: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'For Enterprise Grid workspaces, map local user IDs to global user IDs\n        https://api.slack.com/methods/migration.exchange\n        '
        if isinstance(users, (list, Tuple)):
            kwargs.update({'users': ','.join(users)})
        else:
            kwargs.update({'users': users})
        kwargs.update({'team_id': team_id, 'to_old': to_old})
        return self.api_call('migration.exchange', http_verb='GET', params=kwargs)

    def mpim_close(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Closes a multiparty direct message channel.'
        kwargs.update({'channel': channel})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('mpim.close', json=kwargs)

    def mpim_history(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Fetches history of messages and events from a multiparty direct message.'
        kwargs.update({'channel': channel})
        return self.api_call('mpim.history', http_verb='GET', params=kwargs)

    def mpim_list(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Lists multiparty direct message channels for the calling user.'
        return self.api_call('mpim.list', http_verb='GET', params=kwargs)

    def mpim_mark(self, *, channel: str, ts: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Sets the read cursor in a multiparty direct message channel.'
        kwargs.update({'channel': channel, 'ts': ts})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('mpim.mark', json=kwargs)

    def mpim_open(self, *, users: Union[str, Sequence[str]], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'This method opens a multiparty direct message.'
        if isinstance(users, (list, Tuple)):
            kwargs.update({'users': ','.join(users)})
        else:
            kwargs.update({'users': users})
        return self.api_call('mpim.open', params=kwargs)

    def mpim_replies(self, *, channel: str, thread_ts: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Retrieve a thread of messages posted to a direct message conversation from a\n        multiparty direct message.\n        '
        kwargs.update({'channel': channel, 'thread_ts': thread_ts})
        return self.api_call('mpim.replies', http_verb='GET', params=kwargs)

    def oauth_v2_access(self, *, client_id: str, client_secret: str, code: Optional[str]=None, redirect_uri: Optional[str]=None, grant_type: Optional[str]=None, refresh_token: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Exchanges a temporary OAuth verifier code for an access token.\n        https://api.slack.com/methods/oauth.v2.access\n        '
        if redirect_uri is not None:
            kwargs.update({'redirect_uri': redirect_uri})
        if code is not None:
            kwargs.update({'code': code})
        if grant_type is not None:
            kwargs.update({'grant_type': grant_type})
        if refresh_token is not None:
            kwargs.update({'refresh_token': refresh_token})
        return self.api_call('oauth.v2.access', data=kwargs, auth={'client_id': client_id, 'client_secret': client_secret})

    def oauth_access(self, *, client_id: str, client_secret: str, code: str, redirect_uri: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Exchanges a temporary OAuth verifier code for an access token.\n        https://api.slack.com/methods/oauth.access\n        '
        if redirect_uri is not None:
            kwargs.update({'redirect_uri': redirect_uri})
        kwargs.update({'code': code})
        return self.api_call('oauth.access', data=kwargs, auth={'client_id': client_id, 'client_secret': client_secret})

    def oauth_v2_exchange(self, *, token: str, client_id: str, client_secret: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Exchanges a legacy access token for a new expiring access token and refresh token\n        https://api.slack.com/methods/oauth.v2.exchange\n        '
        kwargs.update({'client_id': client_id, 'client_secret': client_secret, 'token': token})
        return self.api_call('oauth.v2.exchange', params=kwargs)

    def openid_connect_token(self, client_id: str, client_secret: str, code: Optional[str]=None, redirect_uri: Optional[str]=None, grant_type: Optional[str]=None, refresh_token: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Exchanges a temporary OAuth verifier code for an access token for Sign in with Slack.\n        https://api.slack.com/methods/openid.connect.token\n        '
        if redirect_uri is not None:
            kwargs.update({'redirect_uri': redirect_uri})
        if code is not None:
            kwargs.update({'code': code})
        if grant_type is not None:
            kwargs.update({'grant_type': grant_type})
        if refresh_token is not None:
            kwargs.update({'refresh_token': refresh_token})
        return self.api_call('openid.connect.token', data=kwargs, auth={'client_id': client_id, 'client_secret': client_secret})

    def openid_connect_userInfo(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Get the identity of a user who has authorized Sign in with Slack.\n        https://api.slack.com/methods/openid.connect.userInfo\n        '
        return self.api_call('openid.connect.userInfo', params=kwargs)

    def pins_add(self, *, channel: str, timestamp: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Pins an item to a channel.\n        https://api.slack.com/methods/pins.add\n        '
        kwargs.update({'channel': channel, 'timestamp': timestamp})
        return self.api_call('pins.add', params=kwargs)

    def pins_list(self, *, channel: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Lists items pinned to a channel.\n        https://api.slack.com/methods/pins.list\n        '
        kwargs.update({'channel': channel})
        return self.api_call('pins.list', http_verb='GET', params=kwargs)

    def pins_remove(self, *, channel: str, timestamp: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Un-pins an item from a channel.\n        https://api.slack.com/methods/pins.remove\n        '
        kwargs.update({'channel': channel, 'timestamp': timestamp})
        return self.api_call('pins.remove', params=kwargs)

    def reactions_add(self, *, channel: str, name: str, timestamp: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Adds a reaction to an item.\n        https://api.slack.com/methods/reactions.add\n        '
        kwargs.update({'channel': channel, 'name': name, 'timestamp': timestamp})
        return self.api_call('reactions.add', params=kwargs)

    def reactions_get(self, *, channel: Optional[str]=None, file: Optional[str]=None, file_comment: Optional[str]=None, full: Optional[bool]=None, timestamp: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Gets reactions for an item.\n        https://api.slack.com/methods/reactions.get\n        '
        kwargs.update({'channel': channel, 'file': file, 'file_comment': file_comment, 'full': full, 'timestamp': timestamp})
        return self.api_call('reactions.get', http_verb='GET', params=kwargs)

    def reactions_list(self, *, count: Optional[int]=None, cursor: Optional[str]=None, full: Optional[bool]=None, limit: Optional[int]=None, page: Optional[int]=None, team_id: Optional[str]=None, user: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Lists reactions made by a user.\n        https://api.slack.com/methods/reactions.list\n        '
        kwargs.update({'count': count, 'cursor': cursor, 'full': full, 'limit': limit, 'page': page, 'team_id': team_id, 'user': user})
        return self.api_call('reactions.list', http_verb='GET', params=kwargs)

    def reactions_remove(self, *, name: str, channel: Optional[str]=None, file: Optional[str]=None, file_comment: Optional[str]=None, timestamp: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Removes a reaction from an item.\n        https://api.slack.com/methods/reactions.remove\n        '
        kwargs.update({'name': name, 'channel': channel, 'file': file, 'file_comment': file_comment, 'timestamp': timestamp})
        return self.api_call('reactions.remove', params=kwargs)

    def reminders_add(self, *, text: str, time: str, team_id: Optional[str]=None, user: Optional[str]=None, recurrence: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Creates a reminder.\n        https://api.slack.com/methods/reminders.add\n        '
        kwargs.update({'text': text, 'time': time, 'team_id': team_id, 'user': user, 'recurrence': recurrence})
        return self.api_call('reminders.add', params=kwargs)

    def reminders_complete(self, *, reminder: str, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Marks a reminder as complete.\n        https://api.slack.com/methods/reminders.complete\n        '
        kwargs.update({'reminder': reminder, 'team_id': team_id})
        return self.api_call('reminders.complete', params=kwargs)

    def reminders_delete(self, *, reminder: str, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Deletes a reminder.\n        https://api.slack.com/methods/reminders.delete\n        '
        kwargs.update({'reminder': reminder, 'team_id': team_id})
        return self.api_call('reminders.delete', params=kwargs)

    def reminders_info(self, *, reminder: str, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Gets information about a reminder.\n        https://api.slack.com/methods/reminders.info\n        '
        kwargs.update({'reminder': reminder, 'team_id': team_id})
        return self.api_call('reminders.info', http_verb='GET', params=kwargs)

    def reminders_list(self, *, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Lists all reminders created by or for a given user.\n        https://api.slack.com/methods/reminders.list\n        '
        kwargs.update({'team_id': team_id})
        return self.api_call('reminders.list', http_verb='GET', params=kwargs)

    def rtm_connect(self, *, batch_presence_aware: Optional[bool]=None, presence_sub: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Starts a Real Time Messaging session.\n        https://api.slack.com/methods/rtm.connect\n        '
        kwargs.update({'batch_presence_aware': batch_presence_aware, 'presence_sub': presence_sub})
        return self.api_call('rtm.connect', http_verb='GET', params=kwargs)

    def rtm_start(self, *, batch_presence_aware: Optional[bool]=None, include_locale: Optional[bool]=None, mpim_aware: Optional[bool]=None, no_latest: Optional[bool]=None, no_unreads: Optional[bool]=None, presence_sub: Optional[bool]=None, simple_latest: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Starts a Real Time Messaging session.\n        https://api.slack.com/methods/rtm.start\n        '
        kwargs.update({'batch_presence_aware': batch_presence_aware, 'include_locale': include_locale, 'mpim_aware': mpim_aware, 'no_latest': no_latest, 'no_unreads': no_unreads, 'presence_sub': presence_sub, 'simple_latest': simple_latest})
        return self.api_call('rtm.start', http_verb='GET', params=kwargs)

    def search_all(self, *, query: str, count: Optional[int]=None, highlight: Optional[bool]=None, page: Optional[int]=None, sort: Optional[str]=None, sort_dir: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Searches for messages and files matching a query.\n        https://api.slack.com/methods/search.all\n        '
        kwargs.update({'query': query, 'count': count, 'highlight': highlight, 'page': page, 'sort': sort, 'sort_dir': sort_dir, 'team_id': team_id})
        return self.api_call('search.all', http_verb='GET', params=kwargs)

    def search_files(self, *, query: str, count: Optional[int]=None, highlight: Optional[bool]=None, page: Optional[int]=None, sort: Optional[str]=None, sort_dir: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Searches for files matching a query.\n        https://api.slack.com/methods/search.files\n        '
        kwargs.update({'query': query, 'count': count, 'highlight': highlight, 'page': page, 'sort': sort, 'sort_dir': sort_dir, 'team_id': team_id})
        return self.api_call('search.files', http_verb='GET', params=kwargs)

    def search_messages(self, *, query: str, count: Optional[int]=None, cursor: Optional[str]=None, highlight: Optional[bool]=None, page: Optional[int]=None, sort: Optional[str]=None, sort_dir: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Searches for messages matching a query.\n        https://api.slack.com/methods/search.messages\n        '
        kwargs.update({'query': query, 'count': count, 'cursor': cursor, 'highlight': highlight, 'page': page, 'sort': sort, 'sort_dir': sort_dir, 'team_id': team_id})
        return self.api_call('search.messages', http_verb='GET', params=kwargs)

    def stars_add(self, *, channel: Optional[str]=None, file: Optional[str]=None, file_comment: Optional[str]=None, timestamp: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Adds a star to an item.\n        https://api.slack.com/methods/stars.add\n        '
        kwargs.update({'channel': channel, 'file': file, 'file_comment': file_comment, 'timestamp': timestamp})
        return self.api_call('stars.add', params=kwargs)

    def stars_list(self, *, count: Optional[int]=None, cursor: Optional[str]=None, limit: Optional[int]=None, page: Optional[int]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Lists stars for a user.\n        https://api.slack.com/methods/stars.list\n        '
        kwargs.update({'count': count, 'cursor': cursor, 'limit': limit, 'page': page, 'team_id': team_id})
        return self.api_call('stars.list', http_verb='GET', params=kwargs)

    def stars_remove(self, *, channel: Optional[str]=None, file: Optional[str]=None, file_comment: Optional[str]=None, timestamp: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Removes a star from an item.\n        https://api.slack.com/methods/stars.remove\n        '
        kwargs.update({'channel': channel, 'file': file, 'file_comment': file_comment, 'timestamp': timestamp})
        return self.api_call('stars.remove', params=kwargs)

    def team_accessLogs(self, *, before: Optional[Union[int, str]]=None, count: Optional[Union[int, str]]=None, page: Optional[Union[int, str]]=None, team_id: Optional[str]=None, cursor: Optional[str]=None, limit: Optional[int]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Gets the access logs for the current team.\n        https://api.slack.com/methods/team.accessLogs\n        '
        kwargs.update({'before': before, 'count': count, 'page': page, 'team_id': team_id, 'cursor': cursor, 'limit': limit})
        return self.api_call('team.accessLogs', http_verb='GET', params=kwargs)

    def team_billableInfo(self, *, team_id: Optional[str]=None, user: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Gets billable users information for the current team.\n        https://api.slack.com/methods/team.billableInfo\n        '
        kwargs.update({'team_id': team_id, 'user': user})
        return self.api_call('team.billableInfo', http_verb='GET', params=kwargs)

    def team_billing_info(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        "Reads a workspace's billing plan information.\n        https://api.slack.com/methods/team.billing.info\n        "
        return self.api_call('team.billing.info', params=kwargs)

    def team_info(self, *, team: Optional[str]=None, domain: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Gets information about the current team.\n        https://api.slack.com/methods/team.info\n        '
        kwargs.update({'team': team, 'domain': domain})
        return self.api_call('team.info', http_verb='GET', params=kwargs)

    def team_integrationLogs(self, *, app_id: Optional[str]=None, change_type: Optional[str]=None, count: Optional[Union[int, str]]=None, page: Optional[Union[int, str]]=None, service_id: Optional[str]=None, team_id: Optional[str]=None, user: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Gets the integration logs for the current team.\n        https://api.slack.com/methods/team.integrationLogs\n        '
        kwargs.update({'app_id': app_id, 'change_type': change_type, 'count': count, 'page': page, 'service_id': service_id, 'team_id': team_id, 'user': user})
        return self.api_call('team.integrationLogs', http_verb='GET', params=kwargs)

    def team_profile_get(self, *, visibility: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        "Retrieve a team's profile.\n        https://api.slack.com/methods/team.profile.get\n        "
        kwargs.update({'visibility': visibility})
        return self.api_call('team.profile.get', http_verb='GET', params=kwargs)

    def team_preferences_list(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        "Retrieve a list of a workspace's team preferences.\n        https://api.slack.com/methods/team.preferences.list\n        "
        return self.api_call('team.preferences.list', params=kwargs)

    def usergroups_create(self, *, name: str, channels: Optional[Union[str, Sequence[str]]]=None, description: Optional[str]=None, handle: Optional[str]=None, include_count: Optional[bool]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Create a User Group\n        https://api.slack.com/methods/usergroups.create\n        '
        kwargs.update({'name': name, 'description': description, 'handle': handle, 'include_count': include_count, 'team_id': team_id})
        if isinstance(channels, (list, Tuple)):
            kwargs.update({'channels': ','.join(channels)})
        else:
            kwargs.update({'channels': channels})
        return self.api_call('usergroups.create', params=kwargs)

    def usergroups_disable(self, *, usergroup: str, include_count: Optional[bool]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Disable an existing User Group\n        https://api.slack.com/methods/usergroups.disable\n        '
        kwargs.update({'usergroup': usergroup, 'include_count': include_count, 'team_id': team_id})
        return self.api_call('usergroups.disable', params=kwargs)

    def usergroups_enable(self, *, usergroup: str, include_count: Optional[bool]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Enable a User Group\n        https://api.slack.com/methods/usergroups.enable\n        '
        kwargs.update({'usergroup': usergroup, 'include_count': include_count, 'team_id': team_id})
        return self.api_call('usergroups.enable', params=kwargs)

    def usergroups_list(self, *, include_count: Optional[bool]=None, include_disabled: Optional[bool]=None, include_users: Optional[bool]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'List all User Groups for a team\n        https://api.slack.com/methods/usergroups.list\n        '
        kwargs.update({'include_count': include_count, 'include_disabled': include_disabled, 'include_users': include_users, 'team_id': team_id})
        return self.api_call('usergroups.list', http_verb='GET', params=kwargs)

    def usergroups_update(self, *, usergroup: str, channels: Optional[Union[str, Sequence[str]]]=None, description: Optional[str]=None, handle: Optional[str]=None, include_count: Optional[bool]=None, name: Optional[str]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Update an existing User Group\n        https://api.slack.com/methods/usergroups.update\n        '
        kwargs.update({'usergroup': usergroup, 'description': description, 'handle': handle, 'include_count': include_count, 'name': name, 'team_id': team_id})
        if isinstance(channels, (list, Tuple)):
            kwargs.update({'channels': ','.join(channels)})
        else:
            kwargs.update({'channels': channels})
        return self.api_call('usergroups.update', params=kwargs)

    def usergroups_users_list(self, *, usergroup: str, include_disabled: Optional[bool]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'List all users in a User Group\n        https://api.slack.com/methods/usergroups.users.list\n        '
        kwargs.update({'usergroup': usergroup, 'include_disabled': include_disabled, 'team_id': team_id})
        return self.api_call('usergroups.users.list', http_verb='GET', params=kwargs)

    def usergroups_users_update(self, *, usergroup: str, users: Union[str, Sequence[str]], include_count: Optional[bool]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Update the list of users for a User Group\n        https://api.slack.com/methods/usergroups.users.update\n        '
        kwargs.update({'usergroup': usergroup, 'include_count': include_count, 'team_id': team_id})
        if isinstance(users, (list, Tuple)):
            kwargs.update({'users': ','.join(users)})
        else:
            kwargs.update({'users': users})
        return self.api_call('usergroups.users.update', params=kwargs)

    def users_conversations(self, *, cursor: Optional[str]=None, exclude_archived: Optional[bool]=None, limit: Optional[int]=None, team_id: Optional[str]=None, types: Optional[Union[str, Sequence[str]]]=None, user: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'List conversations the calling user may access.\n        https://api.slack.com/methods/users.conversations\n        '
        kwargs.update({'cursor': cursor, 'exclude_archived': exclude_archived, 'limit': limit, 'team_id': team_id, 'user': user})
        if isinstance(types, (list, Tuple)):
            kwargs.update({'types': ','.join(types)})
        else:
            kwargs.update({'types': types})
        return self.api_call('users.conversations', http_verb='GET', params=kwargs)

    def users_deletePhoto(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Delete the user profile photo\n        https://api.slack.com/methods/users.deletePhoto\n        '
        return self.api_call('users.deletePhoto', http_verb='GET', params=kwargs)

    def users_getPresence(self, *, user: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Gets user presence information.\n        https://api.slack.com/methods/users.getPresence\n        '
        kwargs.update({'user': user})
        return self.api_call('users.getPresence', http_verb='GET', params=kwargs)

    def users_identity(self, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        "Get a user's identity.\n        https://api.slack.com/methods/users.identity\n        "
        return self.api_call('users.identity', http_verb='GET', params=kwargs)

    def users_info(self, *, user: str, include_locale: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        'Gets information about a user.\n        https://api.slack.com/methods/users.info\n        '
        kwargs.update({'user': user, 'include_locale': include_locale})
        return self.api_call('users.info', http_verb='GET', params=kwargs)

    def users_list(self, *, cursor: Optional[str]=None, include_locale: Optional[bool]=None, limit: Optional[int]=None, team_id: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        'Lists all users in a Slack team.\n        https://api.slack.com/methods/users.list\n        '
        kwargs.update({'cursor': cursor, 'include_locale': include_locale, 'limit': limit, 'team_id': team_id})
        return self.api_call('users.list', http_verb='GET', params=kwargs)

    def users_lookupByEmail(self, *, email: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        'Find a user with an email address.\n        https://api.slack.com/methods/users.lookupByEmail\n        '
        kwargs.update({'email': email})
        return self.api_call('users.lookupByEmail', http_verb='GET', params=kwargs)

    def users_setPhoto(self, *, image: Union[str, IOBase], crop_w: Optional[Union[int, str]]=None, crop_x: Optional[Union[int, str]]=None, crop_y: Optional[Union[int, str]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            print('Hello World!')
        'Set the user profile photo\n        https://api.slack.com/methods/users.setPhoto\n        '
        kwargs.update({'crop_w': crop_w, 'crop_x': crop_x, 'crop_y': crop_y})
        return self.api_call('users.setPhoto', files={'image': image}, data=kwargs)

    def users_setPresence(self, *, presence: str, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Manually sets user presence.\n        https://api.slack.com/methods/users.setPresence\n        '
        kwargs.update({'presence': presence})
        return self.api_call('users.setPresence', params=kwargs)

    def users_profile_get(self, *, user: Optional[str]=None, include_labels: Optional[bool]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        "Retrieves a user's profile information.\n        https://api.slack.com/methods/users.profile.get\n        "
        kwargs.update({'user': user, 'include_labels': include_labels})
        return self.api_call('users.profile.get', http_verb='GET', params=kwargs)

    def users_profile_set(self, *, name: Optional[str]=None, value: Optional[str]=None, user: Optional[str]=None, profile: Optional[Dict]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Set the profile information for a user.\n        https://api.slack.com/methods/users.profile.set\n        '
        kwargs.update({'name': name, 'profile': profile, 'user': user, 'value': value})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('users.profile.set', json=kwargs)

    def views_open(self, *, trigger_id: str, view: Union[dict, View], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Open a view for a user.\n        https://api.slack.com/methods/views.open\n        See https://api.slack.com/block-kit/surfaces/modals for details.\n        '
        kwargs.update({'trigger_id': trigger_id})
        if isinstance(view, View):
            kwargs.update({'view': view.to_dict()})
        else:
            kwargs.update({'view': view})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('views.open', json=kwargs)

    def views_push(self, *, trigger_id: str, view: Union[dict, View], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Push a view onto the stack of a root view.\n        Push a new view onto the existing view stack by passing a view\n        payload and a valid trigger_id generated from an interaction\n        within the existing modal.\n        Read the modals documentation (https://api.slack.com/block-kit/surfaces/modals)\n        to learn more about the lifecycle and intricacies of views.\n        https://api.slack.com/methods/views.push\n        '
        kwargs.update({'trigger_id': trigger_id})
        if isinstance(view, View):
            kwargs.update({'view': view.to_dict()})
        else:
            kwargs.update({'view': view})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('views.push', json=kwargs)

    def views_update(self, *, view: Union[dict, View], external_id: Optional[str]=None, view_id: Optional[str]=None, hash: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Update an existing view.\n        Update a view by passing a new view definition along with the\n        view_id returned in views.open or the external_id.\n        See the modals documentation (https://api.slack.com/block-kit/surfaces/modals#updating_views)\n        to learn more about updating views and avoiding race conditions with the hash argument.\n        https://api.slack.com/methods/views.update\n        '
        if isinstance(view, View):
            kwargs.update({'view': view.to_dict()})
        else:
            kwargs.update({'view': view})
        if external_id:
            kwargs.update({'external_id': external_id})
        elif view_id:
            kwargs.update({'view_id': view_id})
        else:
            raise e.SlackRequestError('Either view_id or external_id is required.')
        kwargs.update({'hash': hash})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('views.update', json=kwargs)

    def views_publish(self, *, user_id: str, view: Union[dict, View], hash: Optional[str]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            return 10
        "Publish a static view for a User.\n        Create or update the view that comprises an\n        app's Home tab (https://api.slack.com/surfaces/tabs)\n        https://api.slack.com/methods/views.publish\n        "
        kwargs.update({'user_id': user_id, 'hash': hash})
        if isinstance(view, View):
            kwargs.update({'view': view.to_dict()})
        else:
            kwargs.update({'view': view})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('views.publish', json=kwargs)

    def workflows_stepCompleted(self, *, workflow_step_execute_id: str, outputs: Optional[dict]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        "Indicate a successful outcome of a workflow step's execution.\n        https://api.slack.com/methods/workflows.stepCompleted\n        "
        kwargs.update({'workflow_step_execute_id': workflow_step_execute_id})
        if outputs is not None:
            kwargs.update({'outputs': outputs})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('workflows.stepCompleted', json=kwargs)

    def workflows_stepFailed(self, *, workflow_step_execute_id: str, error: Dict[str, str], **kwargs) -> Union[Future, SlackResponse]:
        if False:
            while True:
                i = 10
        "Indicate an unsuccessful outcome of a workflow step's execution.\n        https://api.slack.com/methods/workflows.stepFailed\n        "
        kwargs.update({'workflow_step_execute_id': workflow_step_execute_id, 'error': error})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('workflows.stepFailed', json=kwargs)

    def workflows_updateStep(self, *, workflow_step_edit_id: str, inputs: Optional[Dict[str, Any]]=None, outputs: Optional[List[Dict[str, str]]]=None, **kwargs) -> Union[Future, SlackResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Update the configuration for a workflow extension step.\n        https://api.slack.com/methods/workflows.updateStep\n        '
        kwargs.update({'workflow_step_edit_id': workflow_step_edit_id})
        if inputs is not None:
            kwargs.update({'inputs': inputs})
        if outputs is not None:
            kwargs.update({'outputs': outputs})
        kwargs = _remove_none_values(kwargs)
        return self.api_call('workflows.updateStep', json=kwargs)