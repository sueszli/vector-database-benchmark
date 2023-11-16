from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import control_silo_endpoint
from sentry.api.bases.user import UserEndpoint
from sentry.api.decorators import sudo_required
from sentry.api.paginator import DateTimePaginator
from sentry.api.serializers import serialize
from sentry.models.userip import UserIP

@control_silo_endpoint
class UserIPsEndpoint(UserEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    @sudo_required
    def get(self, request: Request, user) -> Response:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get list of IP addresses\n        ````````````````````````\n\n        Returns a list of IP addresses used to authenticate against this account.\n\n        :auth required:\n        '
        queryset = UserIP.objects.filter(user=user)
        return self.paginate(request=request, queryset=queryset, order_by='-last_seen', paginator_cls=DateTimePaginator, on_results=lambda x: serialize(x, request))