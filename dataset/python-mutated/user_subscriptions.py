from django.db.models import F
from django.utils import timezone
from rest_framework import serializers
from sentry import newsletter
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import control_silo_endpoint
from sentry.api.bases.user import UserEndpoint
from sentry.models.user import User
from sentry.models.useremail import UserEmail

class DefaultNewsletterValidator(serializers.Serializer):
    subscribed = serializers.BooleanField(required=True)

class NewsletterValidator(serializers.Serializer):
    listId = serializers.IntegerField(required=True)
    subscribed = serializers.BooleanField(required=True)
from rest_framework.request import Request
from rest_framework.response import Response

@control_silo_endpoint
class UserSubscriptionsEndpoint(UserEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN, 'PUT': ApiPublishStatus.UNKNOWN, 'POST': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, user) -> Response:
        if False:
            while True:
                i = 10
        '\n        Retrieve Account Subscriptions\n        `````````````````````````````````````\n\n        Return list of subscriptions for an account\n\n        :auth: required\n        '
        sub = newsletter.get_subscriptions(user)
        if sub is None or not newsletter.is_enabled():
            return self.respond([])
        return self.respond([{'listId': x.get('list_id'), 'listDescription': x.get('list_description'), 'listName': x.get('list_name'), 'email': x.get('email'), 'subscribed': x.get('subscribed'), 'subscribedDate': x.get('subscribed_date'), 'unsubscribedDate': x.get('unsubscribed_date')} for x in sub['subscriptions']])

    def put(self, request: Request, user) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Update Account Subscriptions\n        ````````````````````````````\n\n        Update account subscriptions to newsletter\n\n        :param int listId: id of newsletter list\n        :param boolean subscribed: should be subscribed to newsletter\n        :auth: required\n        '
        validator = NewsletterValidator(data=request.data)
        if not validator.is_valid():
            return self.respond(validator.errors, status=400)
        result = validator.validated_data
        email = UserEmail.objects.get_primary_email(user)
        kwargs = {'list_id': result['listId'], 'subscribed': result['subscribed'], 'verified': email.is_verified}
        if not result['subscribed']:
            kwargs['unsubscribed_date'] = timezone.now()
        else:
            kwargs['subscribed_date'] = timezone.now()
        newsletter.create_or_update_subscription(user, **kwargs)
        return self.respond(status=204)

    def post(self, request: Request, user) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Configure Newsletter Subscription\n        `````````````````````````````````\n\n        Update the default newsletter subscription.\n\n        :param boolean subscribed: should be subscribed to newsletter\n        :auth: required\n        '
        validator = DefaultNewsletterValidator(data=request.data)
        if not validator.is_valid():
            return self.respond(validator.errors, status=400)
        result = validator.validated_data
        email = UserEmail.objects.get_primary_email(user)
        kwargs = {'subscribed': result['subscribed'], 'verified': email.is_verified, 'list_ids': newsletter.get_default_list_ids()}
        if not result['subscribed']:
            kwargs['unsubscribed_date'] = timezone.now()
        else:
            kwargs['subscribed_date'] = timezone.now()
        newsletter.create_or_update_subscriptions(user, **kwargs)
        user.update(flags=F('flags').bitand(~User.flags.newsletter_consent_prompt))
        return self.respond(status=204)