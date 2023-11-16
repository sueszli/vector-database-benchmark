from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import control_silo_endpoint
from sentry.api.bases.user import UserEndpoint
from sentry.api.serializers import serialize
from sentry.api.serializers.models.notification_setting import NotificationSettingsSerializer
from sentry.api.validators.notifications import validate, validate_type_option
from sentry.models.notificationsetting import NotificationSetting
from sentry.models.user import User

@control_silo_endpoint
class UserNotificationSettingsDetailsEndpoint(UserEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN, 'PUT': ApiPublishStatus.UNKNOWN}
    '\n    This Notification Settings endpoint is the generic way to interact with the\n    NotificationSettings table via the API.\n    TODO(mgaeta): If this is going to replace the UserNotificationDetailsEndpoint\n     and UserNotificationFineTuningEndpoint endpoints, then it should probably\n     be able to translate legacy values from UserOptions.\n    '

    def get(self, request: Request, user: User) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Get the Notification Settings for a given User.\n        ````````````````````````````````\n        :pparam string user_id: A User\'s `user_id` or "me" for current user.\n        :qparam string type: If set, filter the NotificationSettings to this type.\n\n        :auth required:\n        '
        type_option = validate_type_option(request.GET.get('type'))
        notification_preferences = serialize(user, request.user, NotificationSettingsSerializer(), type=type_option)
        return Response(notification_preferences)

    def put(self, request: Request, user: User) -> Response:
        if False:
            print('Hello World!')
        '\n        Update the Notification Settings for a given User.\n        ````````````````````````````````\n        :pparam string user_id: A User\'s `user_id` or "me" for current user.\n        :param map <anonymous>: The POST data for this request should be several\n            nested JSON mappings. The bottommost value is the "value" of the\n            notification setting and the order of scoping is:\n              - type (str),\n              - scope_type (str),\n              - scope_identifier (int or str)\n              - provider (str)\n            Example: {\n                "workflow": {\n                    "user": {\n                        "me": {\n                            "email": "never",\n                            "slack": "never"\n                        },\n                    },\n                    "project": {\n                        1: {\n                            "email": "always",\n                            "slack": "always"\n                        },\n                        2: {\n                            "email": "subscribe_only",\n                            "slack": "subscribe_only"\n                        }\n                    }\n                }\n            }\n\n        :auth required:\n        '
        notification_settings = validate(request.data, user=user)
        NotificationSetting.objects.update_settings_bulk(notification_settings, user=user)
        return Response(status=status.HTTP_204_NO_CONTENT)