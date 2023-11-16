from django.http import HttpRequest
from sentry.types.activity import ActivityType
from .mail import ActivityMailDebugView

class DebugResolvedInReleaseEmailView(ActivityMailDebugView):

    def get_activity(self, request: HttpRequest, event):
        if False:
            i = 10
            return i + 15
        return {'type': ActivityType.SET_RESOLVED_IN_RELEASE.value, 'data': {'version': 'abcdef'}}

class DebugResolvedInReleaseUpcomingEmailView(ActivityMailDebugView):

    def get_activity(self, request: HttpRequest, event):
        if False:
            print('Hello World!')
        return {'type': ActivityType.SET_RESOLVED_IN_RELEASE.value, 'data': {'version': ''}}