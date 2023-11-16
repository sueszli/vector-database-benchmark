from collections import defaultdict
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Set, Union
from sentry.api.serializers import Serializer
from sentry.models.notificationsetting import NotificationSetting
from sentry.models.team import Team
from sentry.models.user import User
from sentry.notifications.types import NotificationSettingTypes
from sentry.services.hybrid_cloud.organization import RpcTeam
from sentry.services.hybrid_cloud.user import RpcUser

class NotificationSettingsSerializer(Serializer):
    """
    This Serializer fetches and serializes NotificationSettings for a list of
    targets (users or teams.) Pass filters like `project=project` and
    `type=NotificationSettingTypes.DEPLOY` to kwargs.
    """

    def get_attrs(self, item_list: Iterable[Union['Team', 'User']], user: User, **kwargs: Any) -> MutableMapping[Union['Team', 'User'], MutableMapping[str, Set[Any]]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This takes a list of recipients (which are either Users or Teams,\n        because both can have Notification Settings). The function\n        returns a mapping of targets to flat lists of object to be passed to the\n        `serialize` function.\n\n        :param item_list: Either a Set of User or Team objects whose\n            notification settings should be serialized.\n        :param user: The user who will be viewing the notification settings.\n        :param kwargs: Dict of optional filter options:\n            - type: NotificationSettingTypes enum value. e.g. WORKFLOW, DEPLOY.\n        '
        type_option: Optional[NotificationSettingTypes] = kwargs.get('type')
        team_map = {t.id: t for t in item_list if isinstance(t, (Team, RpcTeam))}
        user_map = {u.id: u for u in item_list if isinstance(u, (User, RpcUser))}
        notifications_settings = NotificationSetting.objects._filter(type=type_option, team_ids=list(team_map.keys()), user_ids=list(user_map.keys()))
        result: MutableMapping[Union[Team, User], MutableMapping[str, Set[Any]]] = defaultdict(lambda : defaultdict(set))
        for (_, team) in team_map.items():
            result[team]['settings'] = set()
        for (_, user) in user_map.items():
            result[user]['settings'] = set()
        for notifications_setting in notifications_settings:
            if notifications_setting.user_id:
                target_user = user_map[notifications_setting.user_id]
                result[target_user]['settings'].add(notifications_setting)
            elif notifications_setting.team_id:
                target_team = team_map[notifications_setting.team_id]
                result[target_team]['settings'].add(notifications_setting)
            else:
                raise ValueError(f'NotificationSetting {notifications_setting.id} has neither team_id nor user_id')
        return result

    def serialize(self, obj: Union[Team, User], attrs: Mapping[str, Iterable[Any]], user: User, **kwargs: Any) -> Mapping[str, Mapping[str, Mapping[int, Mapping[str, str]]]]:
        if False:
            return 10
        '\n        Convert a user or team\'s NotificationSettings to a python object\n        comprised of primitives. This will backfill all possible notification\n        settings with the appropriate defaults.\n\n        Example: {\n            "workflow": {\n                "project": {\n                    1: {\n                        "email": "always",\n                        "slack": "always"\n                    },\n                    2: {\n                        "email": "subscribe_only",\n                        "slack": "subscribe_only"\n                    }\n                }\n            }\n        }\n\n        :param obj: A user or team.\n        :param attrs: The `obj` target\'s NotificationSettings\n        :param user: The user who will be viewing the NotificationSettings.\n        :param kwargs: The same `kwargs` as `get_attrs`.\n        :returns A mapping. See example.\n        '
        data: MutableMapping[str, MutableMapping[str, MutableMapping[int, MutableMapping[str, str]]]] = defaultdict(lambda : defaultdict(lambda : defaultdict(dict)))
        for n in attrs['settings']:
            data[n.type_str][n.scope_str][n.scope_identifier][n.provider_str] = n.value_str
        return data