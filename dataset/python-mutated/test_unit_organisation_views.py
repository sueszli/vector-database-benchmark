import json
from datetime import datetime, timedelta
from typing import Type
from unittest import TestCase, mock
from unittest.mock import MagicMock
import pytest
from _pytest.logging import LogCaptureFixture
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core import mail
from django.db.models import Model
from django.urls import reverse
from freezegun import freeze_time
from pytz import UTC
from rest_framework import status
from rest_framework.test import APIClient, override_settings
from environments.models import Environment
from environments.permissions.models import UserEnvironmentPermission
from features.models import Feature, FeatureSegment
from organisations.chargebee.metadata import ChargebeeObjMetadata
from organisations.invites.models import Invite
from organisations.models import Organisation, OrganisationRole, OrganisationSubscriptionInformationCache, OrganisationWebhook, Subscription
from organisations.permissions.models import UserOrganisationPermission
from organisations.permissions.permissions import CREATE_PROJECT
from organisations.subscriptions.constants import CHARGEBEE, MAX_API_CALLS_IN_FREE_PLAN, MAX_PROJECTS_IN_FREE_PLAN, MAX_SEATS_IN_FREE_PLAN
from projects.models import Project, UserProjectPermission
from segments.models import Segment
from users.models import FFAdminUser, UserPermissionGroup, UserPermissionGroupMembership
from util.tests import Helper
User = get_user_model()

@pytest.mark.django_db
class OrganisationTestCase(TestCase):
    post_template = '{ "name" : "%s", "webhook_notification_email": "%s" }'
    put_template = '{ "name" : "%s"}'

    def setUp(self):
        if False:
            while True:
                i = 10
        self.client = APIClient()
        self.user = Helper.create_ffadminuser()
        self.client.force_authenticate(user=self.user)

    def test_should_return_organisation_list_when_requested(self):
        if False:
            print('Hello World!')
        organisation = Organisation.objects.create(name='Test org')
        self.user.add_organisation(organisation)
        response = self.client.get('/api/v1/organisations/')
        assert response.status_code == status.HTTP_200_OK
        assert 'count' in response.data and response.data['count'] == 1
        response_json = response.json()
        org_data = response_json['results'][0]
        assert 'persist_trait_data' in org_data

    def test_non_superuser_can_create_new_organisation_by_default(self):
        if False:
            for i in range(10):
                print('nop')
        user = User.objects.create(email='test@example.com')
        client = APIClient()
        client.force_authenticate(user=user)
        org_name = 'Test create org'
        webhook_notification_email = 'test@email.com'
        url = reverse('api-v1:organisations:organisation-list')
        data = {'name': org_name, 'webhook_notification_email': webhook_notification_email}
        response = client.post(url, data=data)
        assert response.status_code == status.HTTP_201_CREATED
        assert Organisation.objects.get(name=org_name).webhook_notification_email == webhook_notification_email

    @override_settings(RESTRICT_ORG_CREATE_TO_SUPERUSERS=True)
    def test_create_new_orgnisation_returns_403_with_non_superuser(self):
        if False:
            print('Hello World!')
        user = User.objects.create(email='test@example.com')
        client = APIClient()
        client.force_authenticate(user=user)
        org_name = 'Test create org'
        url = reverse('api-v1:organisations:organisation-list')
        data = {'name': org_name}
        response = client.post(url, data=data)
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'You do not have permission to perform this action.' == response.json()['detail']

    def test_should_update_organisation_data(self):
        if False:
            for i in range(10):
                print('nop')
        original_organisation_name = 'test org'
        new_organisation_name = 'new test org'
        organisation = Organisation.objects.create(name=original_organisation_name)
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        url = reverse('api-v1:organisations:organisation-detail', args=[organisation.pk])
        data = {'name': new_organisation_name, 'restrict_project_create_to_admin': True}
        response = self.client.put(url, data=data)
        organisation.refresh_from_db()
        assert response.status_code == status.HTTP_200_OK
        assert organisation.name == new_organisation_name
        assert organisation.restrict_project_create_to_admin

    @override_settings()
    def test_should_invite_users(self):
        if False:
            while True:
                i = 10
        settings.REST_FRAMEWORK['DEFAULT_THROTTLE_RATES']['invite'] = None
        org_name = 'test_org'
        organisation = Organisation.objects.create(name=org_name)
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        url = reverse('api-v1:organisations:organisation-invite', args=[organisation.pk])
        data = {'emails': ['test@example.com']}
        response = self.client.post(url, data=json.dumps(data), content_type='application/json')
        assert response.status_code == status.HTTP_201_CREATED
        assert Invite.objects.filter(email='test@example.com').exists()

    @override_settings()
    def test_should_fail_if_invite_exists_already(self):
        if False:
            while True:
                i = 10
        settings.REST_FRAMEWORK['DEFAULT_THROTTLE_RATES']['invite'] = None
        organisation = Organisation.objects.create(name='test org')
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        email = 'test_2@example.com'
        data = {'emails': [email]}
        url = reverse('api-v1:organisations:organisation-invite', args=[organisation.pk])
        response_success = self.client.post(url, data=json.dumps(data), content_type='application/json')
        response_fail = self.client.post(url, data=json.dumps(data), content_type='application/json')
        assert response_success.status_code == status.HTTP_201_CREATED
        assert response_fail.status_code == status.HTTP_400_BAD_REQUEST
        assert Invite.objects.filter(email=email, organisation=organisation).count() == 1

    @override_settings()
    def test_should_return_all_invites_and_can_resend(self):
        if False:
            return 10
        settings.REST_FRAMEWORK['DEFAULT_THROTTLE_RATES']['invite'] = None
        organisation = Organisation.objects.create(name='Test org 2')
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        invite_1 = Invite.objects.create(email='test_1@example.com', organisation=organisation)
        Invite.objects.create(email='test_2@example.com', organisation=organisation)
        invite_list_response = self.client.get('/api/v1/organisations/%s/invites/' % organisation.id)
        invite_resend_response = self.client.post('/api/v1/organisations/%s/invites/%s/resend/' % (organisation.id, invite_1.id))
        assert invite_list_response.status_code == status.HTTP_200_OK
        assert invite_resend_response.status_code == status.HTTP_200_OK

    def test_remove_user_from_an_organisation_also_removes_from_group(self):
        if False:
            for i in range(10):
                print('nop')
        organisation = Organisation.objects.create(name='Test org')
        group = UserPermissionGroup.objects.create(name='Test Group', organisation=organisation)
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        user_2 = FFAdminUser.objects.create(email='test@example.com')
        user_2.add_organisation(organisation)
        group.users.add(user_2)
        group.users.add(self.user)
        url = reverse('api-v1:organisations:organisation-remove-users', args=[organisation.pk])
        data = [{'id': user_2.pk}]
        res = self.client.post(url, data=json.dumps(data), content_type='application/json')
        assert res.status_code == status.HTTP_200_OK
        assert organisation not in user_2.organisations.all()
        assert group not in user_2.permission_groups.all()
        assert group in self.user.permission_groups.all()

    def test_remove_user_from_an_organisation_also_removes_users_environment_and_project_permission(self):
        if False:
            print('Hello World!')
        organisation = Organisation.objects.create(name='Test org')
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        project_name = 'org_remove_test'
        project_create_url = reverse('api-v1:projects:project-list')
        data = {'name': project_name, 'organisation': organisation.id}
        response = self.client.post(project_create_url, data=data)
        project_id = response.json()['id']
        url = reverse('api-v1:environments:environment-list')
        data = {'name': 'Test environment', 'project': project_id}
        response = self.client.post(url, data=data)
        environment_id = response.json()['id']
        url = reverse('api-v1:organisations:organisation-remove-users', args=[organisation.pk])
        data = [{'id': self.user.id}]
        res = self.client.post(url, data=json.dumps(data), content_type='application/json')
        assert res.status_code == status.HTTP_200_OK
        assert UserProjectPermission.objects.filter(project__id=project_id, user=self.user).count() == 0
        assert UserEnvironmentPermission.objects.filter(user=self.user, environment__id=environment_id).count() == 0

    @override_settings()
    def test_can_invite_user_as_admin(self):
        if False:
            print('Hello World!')
        settings.REST_FRAMEWORK['DEFAULT_THROTTLE_RATES']['invite'] = None
        organisation = Organisation.objects.create(name='Test org')
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        url = reverse('api-v1:organisations:organisation-invite', args=[organisation.pk])
        invited_email = 'test@example.com'
        data = {'invites': [{'email': invited_email, 'role': OrganisationRole.ADMIN.name}]}
        self.client.post(url, data=json.dumps(data), content_type='application/json')
        assert Invite.objects.filter(email=invited_email).exists()
        assert Invite.objects.get(email=invited_email).role == OrganisationRole.ADMIN.name

    @override_settings()
    def test_can_invite_user_as_user(self):
        if False:
            print('Hello World!')
        settings.REST_FRAMEWORK['DEFAULT_THROTTLE_RATES']['invite'] = None
        organisation = Organisation.objects.create(name='Test org')
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        url = reverse('api-v1:organisations:organisation-invite', args=[organisation.pk])
        invited_email = 'test@example.com'
        data = {'invites': [{'email': invited_email, 'role': OrganisationRole.USER.name}]}
        self.client.post(url, data=json.dumps(data), content_type='application/json')
        assert Invite.objects.filter(email=invited_email).exists()
        assert Invite.objects.get(email=invited_email).role == OrganisationRole.USER.name

    def test_user_can_get_projects_for_an_organisation(self):
        if False:
            for i in range(10):
                print('nop')
        organisation = Organisation.objects.create(name='Test org')
        self.user.add_organisation(organisation, OrganisationRole.USER)
        url = reverse('api-v1:organisations:organisation-projects', args=[organisation.pk])
        res = self.client.get(url)
        assert res.status_code == status.HTTP_200_OK

    @mock.patch('app_analytics.influxdb_wrapper.influxdb_client')
    def test_should_get_usage_for_organisation(self, mock_influxdb_client):
        if False:
            i = 10
            return i + 15
        org_name = 'test_org'
        organisation = Organisation.objects.create(name=org_name)
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        url = reverse('api-v1:organisations:organisation-usage', args=[organisation.pk])
        influx_org = settings.INFLUXDB_ORG
        read_bucket = settings.INFLUXDB_BUCKET + '_downsampled_15m'
        expected_query = f'from(bucket:"{read_bucket}") |> range(start: -30d, stop: now()) |> filter(fn:(r) => r._measurement == "api_call")         |> filter(fn: (r) => r["_field"] == "request_count")         |> filter(fn: (r) => r["organisation_id"] == "{organisation.id}") |> drop(columns: ["organisation", "project", "project_id", "environment", "environment_id"])|> sum()'.replace(' ', '').replace('\n', '')
        response = self.client.get(url, content_type='application/json')
        assert response.status_code == status.HTTP_200_OK
        mock_influxdb_client.query_api.return_value.query.assert_called_once()
        call = mock_influxdb_client.query_api.return_value.query.mock_calls[0]
        assert call[2]['org'] == influx_org
        assert call[2]['query'].replace(' ', '').replace('\n', '') == expected_query

    @override_settings(ENABLE_CHARGEBEE=True)
    @mock.patch('organisations.serializers.get_subscription_data_from_hosted_page')
    def test_update_subscription_gets_subscription_data_from_chargebee(self, mock_get_subscription_data):
        if False:
            for i in range(10):
                print('nop')
        organisation = Organisation.objects.create(name='Test org')
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        url = reverse('api-v1:organisations:organisation-update-subscription', args=[organisation.pk])
        hosted_page_id = 'some-id'
        data = {'hosted_page_id': hosted_page_id}
        customer_id = 'customer-id'
        subscription_id = 'subscription-id'
        mock_get_subscription_data.return_value = {'subscription_id': subscription_id, 'plan': 'plan-id', 'max_seats': 3, 'subscription_date': datetime.now(tz=UTC), 'customer_id': customer_id}
        res = self.client.post(url, data=data)
        assert res.status_code == status.HTTP_200_OK
        organisation.refresh_from_db()
        mock_get_subscription_data.assert_called_with(hosted_page_id=hosted_page_id)
        assert organisation.has_subscription() and organisation.subscription.subscription_id == subscription_id and (organisation.subscription.customer_id == customer_id)

    def test_delete_organisation(self):
        if False:
            print('Hello World!')
        organisation = Organisation.objects.create(name='Test organisation')
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        project = Project.objects.create(name='Test project', organisation=organisation)
        environment = Environment.objects.create(name='Test environment', project=project)
        feature = Feature.objects.create(name='Test feature', project=project)
        segment = Segment.objects.create(name='Test segment', project=project)
        FeatureSegment.objects.create(feature=feature, segment=segment, environment=environment)
        delete_organisation_url = reverse('api-v1:organisations:organisation-detail', args=[organisation.id])
        response = self.client.delete(delete_organisation_url)
        assert response.status_code == status.HTTP_204_NO_CONTENT

    @mock.patch('organisations.serializers.get_hosted_page_url_for_subscription_upgrade')
    def test_get_hosted_page_url_for_subscription_upgrade(self, mock_get_hosted_page_url):
        if False:
            print('Hello World!')
        organisation = Organisation.objects.create(name='Test organisation')
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        subscription = Subscription.objects.get(organisation=organisation)
        subscription.subscription_id = 'sub-id'
        subscription.save()
        url = reverse('api-v1:organisations:organisation-get-hosted-page-url-for-subscription-upgrade', args=[organisation.id])
        expected_url = 'https://some.url.com/hosted/page'
        mock_get_hosted_page_url.return_value = expected_url
        plan_id = 'plan-id'
        response = self.client.post(url, data=json.dumps({'plan_id': plan_id}), content_type='application/json')
        assert response.status_code == status.HTTP_200_OK
        assert response.json()['url'] == expected_url
        mock_get_hosted_page_url.assert_called_once_with(subscription_id=subscription.subscription_id, plan_id=plan_id)

    def test_get_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        url = reverse('api-v1:organisations:organisation-permissions')
        response = self.client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2

    def test_get_my_permissions_for_non_admin(self):
        if False:
            return 10
        organisation = Organisation.objects.create(name='Test org')
        self.user.add_organisation(organisation)
        user_permission = UserOrganisationPermission.objects.create(user=self.user, organisation=organisation)
        user_permission.add_permission(CREATE_PROJECT)
        url = reverse('api-v1:organisations:organisation-my-permissions', args=[organisation.id])
        response = self.client.get(url)
        assert response.status_code == status.HTTP_200_OK
        response_json = response.json()
        assert response_json['permissions'] == [CREATE_PROJECT]
        assert response_json['admin'] is False

    def test_get_my_permissions_for_admin(self):
        if False:
            while True:
                i = 10
        organisation = Organisation.objects.create(name='Test org')
        self.user.add_organisation(organisation, OrganisationRole.ADMIN)
        url = reverse('api-v1:organisations:organisation-my-permissions', args=[organisation.id])
        response = self.client.get(url)
        assert response.status_code == status.HTTP_200_OK
        response_json = response.json()
        assert response_json['permissions'] == []
        assert response_json['admin'] is True

@pytest.mark.django_db
class ChargeBeeWebhookTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.client = APIClient()
        self.cb_user = User.objects.create(email='chargebee@bullet-train.io', username='chargebee')
        self.admin_user = User.objects.create(email='admin@bullet-train.io', username='admin', is_staff=True)
        self.client.force_authenticate(self.cb_user)
        self.organisation = Organisation.objects.create(name='Test org')
        self.url = reverse('api-v1:chargebee-webhook')
        self.subscription_id = 'subscription-id'
        self.old_plan_id = 'old-plan-id'
        self.old_max_seats = 1
        Subscription.objects.filter(organisation=self.organisation).update(organisation=self.organisation, subscription_id=self.subscription_id, plan=self.old_plan_id, max_seats=self.old_max_seats)
        self.subscription = Subscription.objects.get(organisation=self.organisation)

    @mock.patch('organisations.views.extract_subscription_metadata')
    def test_chargebee_webhook(self, mock_extract_subscription_metadata: MagicMock) -> None:
        if False:
            return 10
        seats = 3
        api_calls = 100
        mock_extract_subscription_metadata.return_value = ChargebeeObjMetadata(seats=seats, api_calls=api_calls, projects=None, chargebee_email=self.cb_user.email)
        data = {'content': {'subscription': {'status': 'active', 'id': self.subscription_id}, 'customer': {'email': self.cb_user.email}}}
        response = self.client.post(self.url, data=json.dumps(data), content_type='application/json')
        assert response.status_code == status.HTTP_200_OK
        self.subscription.refresh_from_db()
        subscription_cache = OrganisationSubscriptionInformationCache.objects.get(organisation=self.subscription.organisation)
        assert subscription_cache.allowed_projects is None
        assert subscription_cache.allowed_30d_api_calls == api_calls
        assert subscription_cache.allowed_seats == seats

    @mock.patch('organisations.models.cancel_chargebee_subscription')
    def test_when_subscription_is_set_to_non_renewing_then_cancellation_date_set_and_alert_sent(self, mocked_cancel_chargebee_subscription):
        if False:
            return 10
        cancellation_date = datetime.now(tz=UTC) + timedelta(days=1)
        data = {'content': {'subscription': {'status': 'non_renewing', 'id': self.subscription_id, 'current_term_end': datetime.timestamp(cancellation_date)}, 'customer': {'email': self.cb_user.email}}}
        self.client.post(self.url, data=json.dumps(data), content_type='application/json')
        self.subscription.refresh_from_db()
        assert self.subscription.cancellation_date == cancellation_date
        assert len(mail.outbox) == 1
        mocked_cancel_chargebee_subscription.assert_not_called()

    def test_when_subscription_is_cancelled_then_cancellation_date_set_and_alert_sent(self):
        if False:
            print('Hello World!')
        cancellation_date = datetime.now(tz=UTC) + timedelta(days=1)
        data = {'content': {'subscription': {'status': 'cancelled', 'id': self.subscription_id, 'current_term_end': datetime.timestamp(cancellation_date)}, 'customer': {'email': self.cb_user.email}}}
        self.client.post(self.url, data=json.dumps(data), content_type='application/json')
        self.subscription.refresh_from_db()
        assert self.subscription.cancellation_date == cancellation_date
        assert len(mail.outbox) == 1

    @mock.patch('organisations.views.extract_subscription_metadata')
    def test_when_cancelled_subscription_is_renewed_then_subscription_activated_and_no_cancellation_email_sent(self, mock_extract_subscription_metadata):
        if False:
            while True:
                i = 10
        self.subscription.cancellation_date = datetime.now(tz=UTC) - timedelta(days=1)
        self.subscription.save()
        mail.outbox.clear()
        mock_extract_subscription_metadata.return_value = ChargebeeObjMetadata(seats=3, api_calls=100, projects=1, chargebee_email=self.cb_user.email)
        data = {'content': {'subscription': {'status': 'active', 'id': self.subscription_id}, 'customer': {'email': self.cb_user.email}}}
        self.client.post(self.url, data=json.dumps(data), content_type='application/json')
        self.subscription.refresh_from_db()
        assert not self.subscription.cancellation_date
        assert not mail.outbox

def test_when_chargebee_webhook_received_with_unknown_subscription_id_then_200(api_client: APIClient, caplog: LogCaptureFixture, django_user_model: Type[Model]):
    if False:
        for i in range(10):
            print('nop')
    subscription_id = 'some-random-id'
    cb_user = django_user_model.objects.create(email='test@example.com', is_staff=True)
    api_client.force_authenticate(cb_user)
    data = {'content': {'subscription': {'status': 'active', 'id': subscription_id}, 'customer': {'email': cb_user.email}}}
    url = reverse('api-v1:chargebee-webhook')
    res = api_client.post(url, data=json.dumps(data), content_type='application/json')
    assert res.status_code == status.HTTP_200_OK
    assert len(caplog.records) == 1
    assert caplog.record_tuples[0] == ('organisations.views', 30, f"Couldn't get unique subscription for ChargeBee id {subscription_id}")

@pytest.mark.django_db
class OrganisationWebhookViewSetTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.organisation = Organisation.objects.create(name='Test org')
        self.user = FFAdminUser.objects.create(email='test@test.com')
        self.user.add_organisation(self.organisation, OrganisationRole.ADMIN)
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.list_url = reverse('api-v1:organisations:organisation-webhooks-list', args=[self.organisation.id])
        self.valid_webhook_url = 'http://my.webhook.com/webhooks'

    def test_user_can_create_new_webhook(self):
        if False:
            print('Hello World!')
        data = {'url': self.valid_webhook_url}
        response = self.client.post(self.list_url, data=data)
        assert response.status_code == status.HTTP_201_CREATED

    def test_can_update_secret(self):
        if False:
            while True:
                i = 10
        webhook = OrganisationWebhook.objects.create(url=self.valid_webhook_url, organisation=self.organisation)
        url = reverse('api-v1:organisations:organisation-webhooks-detail', args=[self.organisation.id, webhook.id])
        data = {'secret': 'random_key'}
        res = self.client.patch(url, data=json.dumps(data), content_type='application/json')
        assert res.status_code == status.HTTP_200_OK
        assert res.json()['secret'] == data['secret']
        webhook.refresh_from_db()
        assert webhook.secret == data['secret']

    @mock.patch('webhooks.mixins.trigger_sample_webhook')
    def test_trigger_sample_webhook_calls_trigger_sample_webhook_method_with_correct_arguments(self, trigger_sample_webhook):
        if False:
            i = 10
            return i + 15
        mocked_response = mock.MagicMock(status_code=200)
        trigger_sample_webhook.return_value = mocked_response
        url = reverse('api-v1:organisations:organisation-webhooks-trigger-sample-webhook', args=[self.organisation.id])
        data = {'url': self.valid_webhook_url}
        response = self.client.post(url, data)
        assert response.json()['message'] == 'Request returned 200'
        assert response.status_code == status.HTTP_200_OK
        (args, _) = trigger_sample_webhook.call_args
        assert args[0].url == self.valid_webhook_url

def test_get_subscription_metadata_when_subscription_information_cache_exist(organisation, admin_client, chargebee_subscription):
    if False:
        while True:
            i = 10
    expected_seats = 10
    expected_projects = 5
    expected_projects = 3
    expected_api_calls = 100
    expected_chargebee_email = 'test@example.com'
    OrganisationSubscriptionInformationCache.objects.create(organisation=organisation, allowed_seats=expected_seats, allowed_projects=expected_projects, allowed_30d_api_calls=expected_api_calls, chargebee_email=expected_chargebee_email)
    url = reverse('api-v1:organisations:organisation-get-subscription-metadata', args=[organisation.pk])
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {'max_seats': expected_seats, 'max_projects': expected_projects, 'max_api_calls': expected_api_calls, 'payment_source': CHARGEBEE, 'chargebee_email': expected_chargebee_email}

def test_get_subscription_metadata_when_subscription_information_cache_does_not_exist(mocker, organisation, admin_client, chargebee_subscription):
    if False:
        i = 10
        return i + 15
    expected_seats = 10
    expected_projects = 5
    expected_api_calls = 100
    expected_chargebee_email = 'test@example.com'
    get_subscription_metadata = mocker.patch('organisations.models.get_subscription_metadata_from_id', return_value=ChargebeeObjMetadata(seats=expected_seats, projects=expected_projects, api_calls=expected_api_calls, chargebee_email=expected_chargebee_email))
    url = reverse('api-v1:organisations:organisation-get-subscription-metadata', args=[organisation.pk])
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {'max_seats': expected_seats, 'max_projects': expected_projects, 'max_api_calls': expected_api_calls, 'payment_source': CHARGEBEE, 'chargebee_email': expected_chargebee_email}
    get_subscription_metadata.assert_called_once_with(chargebee_subscription.subscription_id)

def test_get_subscription_metadata_returns_404_if_the_organisation_have_no_subscription(mocker, organisation, admin_client):
    if False:
        i = 10
        return i + 15
    get_subscription_metadata = mocker.patch('organisations.models.get_subscription_metadata_from_id')
    url = reverse('api-v1:organisations:organisation-get-subscription-metadata', args=[organisation.pk])
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_404_NOT_FOUND
    get_subscription_metadata.assert_not_called()

def test_get_subscription_metadata_returns_defaults_if_chargebee_error(organisation, admin_client, chargebee_subscription):
    if False:
        return 10
    url = reverse('api-v1:organisations:organisation-get-subscription-metadata', args=[organisation.pk])
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {'max_seats': MAX_SEATS_IN_FREE_PLAN, 'max_api_calls': MAX_API_CALLS_IN_FREE_PLAN, 'max_projects': MAX_PROJECTS_IN_FREE_PLAN, 'payment_source': None, 'chargebee_email': None}

def test_can_invite_user_with_permission_groups(settings, admin_client, organisation, user_permission_group):
    if False:
        return 10
    settings.REST_FRAMEWORK['DEFAULT_THROTTLE_RATES']['invite'] = None
    url = reverse('api-v1:organisations:organisation-invite', args=[organisation.pk])
    invited_email = 'test@example.com'
    data = {'invites': [{'email': invited_email, 'role': OrganisationRole.ADMIN.name, 'permission_groups': [user_permission_group.id]}]}
    response = admin_client.post(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()[0]['permission_groups'] == [user_permission_group.id]
    invite = Invite.objects.get(email=invited_email)
    assert user_permission_group in invite.permission_groups.all()

@pytest.mark.parametrize('query_string, expected_filter_args', (('', {}), ('project_id=1', {'project_id': 1}), ('project_id=1&environment_id=1', {'project_id': 1, 'environment_id': 1}), ('environment_id=1', {'environment_id': 1})))
def test_organisation_get_influx_data(mocker, admin_client, organisation, query_string, expected_filter_args):
    if False:
        for i in range(10):
            print('nop')
    base_url = reverse('api-v1:organisations:organisation-get-influx-data', args=[organisation.id])
    url = f'{base_url}?{query_string}'
    mock_get_multiple_event_list_for_organisation = mocker.patch('organisations.views.get_multiple_event_list_for_organisation')
    mock_get_multiple_event_list_for_organisation.return_value = []
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    mock_get_multiple_event_list_for_organisation.assert_called_once_with(str(organisation.id), **expected_filter_args)
    assert response.json() == {'events_list': []}

@freeze_time('2023-07-31 12:00:00')
@pytest.mark.parametrize('plan_id, max_seats, max_api_calls, max_projects, is_updated', [('plan-id', 3, 100, 3, False), ('updated-plan-id', 5, 500, 10, True)])
@mock.patch('organisations.models.get_plan_meta_data')
@mock.patch('organisations.views.extract_subscription_metadata')
def test_when_plan_is_changed_max_seats_and_max_api_calls_are_updated(mock_extract_subscription_metadata, mock_get_plan_meta_data, subscription, admin_client, organisation, plan_id, max_seats, max_api_calls, max_projects, is_updated):
    if False:
        i = 10
        return i + 15
    chargebee_email = 'chargebee@test.com'
    url = reverse('api-v1:chargebee-webhook')
    updated_at = datetime.now(tz=UTC) - timedelta(days=1)
    mock_get_plan_meta_data.return_value = {'seats': max_seats, 'api_calls': max_api_calls}
    mock_extract_subscription_metadata.return_value = ChargebeeObjMetadata(seats=max_seats, api_calls=max_api_calls, projects=max_projects, chargebee_email=chargebee_email)
    data = {'content': {'subscription': {'status': 'active', 'id': subscription.subscription_id, 'plan_id': plan_id}, 'customer': {'email': chargebee_email}}}
    if is_updated:
        subscription_information_cache = OrganisationSubscriptionInformationCache.objects.create(organisation=organisation, allowed_seats=1, allowed_30d_api_calls=10, allowed_projects=1, chargebee_email=chargebee_email, chargebee_updated_at=updated_at, influx_updated_at=None)
    res = admin_client.post(url, data=json.dumps(data), content_type='application/json')
    subscription_information_cache = OrganisationSubscriptionInformationCache.objects.get(organisation=organisation)
    subscription.refresh_from_db()
    assert res.status_code == status.HTTP_200_OK
    assert subscription.plan == plan_id
    assert subscription.max_seats == max_seats
    assert subscription.max_api_calls == max_api_calls
    assert subscription_information_cache.allowed_seats == max_seats
    assert subscription_information_cache.allowed_30d_api_calls == max_api_calls
    assert subscription_information_cache.allowed_projects == max_projects
    assert subscription_information_cache.chargebee_email == chargebee_email
    assert subscription_information_cache.chargebee_updated_at
    assert subscription_information_cache.influx_updated_at is None
    if is_updated:
        assert subscription_information_cache.chargebee_updated_at > updated_at

def test_delete_organisation_does_not_delete_all_subscriptions_from_the_database(admin_client, admin_user, organisation, subscription):
    if False:
        return 10
    '\n    Test to verify workaround for bug in django-softdelete as per issue here:\n    https://github.com/scoursen/django-softdelete/issues/99\n    '
    another_organisation = Organisation.objects.create(name='another org')
    admin_user.add_organisation(another_organisation)
    url = reverse('api-v1:organisations:organisation-detail', args=[organisation.id])
    response = admin_client.delete(url)
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert Subscription.objects.filter(organisation=another_organisation).exists()

def test_make_user_group_admin_user_does_not_belong_to_group(admin_client, admin_user, organisation, user_permission_group):
    if False:
        while True:
            i = 10
    another_user = FFAdminUser.objects.create(email='another_user@example.com')
    another_user.add_organisation(organisation)
    url = reverse('api-v1:organisations:make-user-group-admin', args=[organisation.id, user_permission_group.id, another_user.id])
    response = admin_client.post(url)
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_make_user_group_admin_success(admin_client, admin_user, organisation, user_permission_group):
    if False:
        while True:
            i = 10
    another_user = FFAdminUser.objects.create(email='another_user@example.com')
    another_user.add_organisation(organisation)
    another_user.permission_groups.add(user_permission_group)
    url = reverse('api-v1:organisations:make-user-group-admin', args=[organisation.id, user_permission_group.id, another_user.id])
    response = admin_client.post(url)
    assert response.status_code == status.HTTP_200_OK
    assert UserPermissionGroupMembership.objects.get(ffadminuser=another_user, userpermissiongroup=user_permission_group).group_admin is True

def test_make_user_group_admin_forbidden(staff_client: FFAdminUser, organisation: Organisation, user_permission_group: UserPermissionGroup):
    if False:
        return 10
    another_user = FFAdminUser.objects.create(email='another_user@example.com')
    another_user.add_organisation(organisation)
    another_user.permission_groups.add(user_permission_group)
    url = reverse('api-v1:organisations:make-user-group-admin', args=[organisation.id, user_permission_group.id, another_user.id])
    response = staff_client.post(url)
    assert response.status_code == status.HTTP_403_FORBIDDEN

def test_remove_user_as_group_admin_user_does_not_belong_to_group(admin_client, admin_user, organisation, user_permission_group):
    if False:
        i = 10
        return i + 15
    another_user = FFAdminUser.objects.create(email='another_user@example.com')
    another_user.add_organisation(organisation)
    url = reverse('api-v1:organisations:remove-user-group-admin', args=[organisation.id, user_permission_group.id, another_user.id])
    response = admin_client.post(url)
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_remove_user_as_group_admin_success(admin_client, admin_user, organisation, user_permission_group):
    if False:
        print('Hello World!')
    another_user = FFAdminUser.objects.create(email='another_user@example.com')
    another_user.add_organisation(organisation)
    another_user.permission_groups.add(user_permission_group)
    another_user.make_group_admin(user_permission_group.id)
    url = reverse('api-v1:organisations:remove-user-group-admin', args=[organisation.id, user_permission_group.id, another_user.id])
    response = admin_client.post(url)
    assert response.status_code == status.HTTP_200_OK
    assert UserPermissionGroupMembership.objects.get(ffadminuser=another_user, userpermissiongroup=user_permission_group).group_admin is False

def test_remove_user_as_group_admin_forbidden(staff_client: FFAdminUser, organisation: Organisation, user_permission_group: UserPermissionGroup):
    if False:
        i = 10
        return i + 15
    another_user = FFAdminUser.objects.create(email='another_user@example.com')
    another_user.add_organisation(organisation)
    another_user.permission_groups.add(user_permission_group)
    another_user.make_group_admin(user_permission_group.id)
    url = reverse('api-v1:organisations:remove-user-group-admin', args=[organisation.id, user_permission_group.id, another_user.id])
    response = staff_client.post(url)
    assert response.status_code == status.HTTP_403_FORBIDDEN

def test_list_user_groups_as_group_admin(organisation, api_client):
    if False:
        while True:
            i = 10
    user1 = FFAdminUser.objects.create(email='user1@example.com')
    user2 = FFAdminUser.objects.create(email='user2@example.com')
    user1.add_organisation(organisation)
    user2.add_organisation(organisation)
    user_permission_group_1 = UserPermissionGroup.objects.create(organisation=organisation, name='group1')
    user_permission_group_2 = UserPermissionGroup.objects.create(organisation=organisation, name='group2')
    UserPermissionGroupMembership.objects.create(ffadminuser=user1, userpermissiongroup=user_permission_group_1, group_admin=True)
    UserPermissionGroupMembership.objects.create(ffadminuser=user2, userpermissiongroup=user_permission_group_2, group_admin=True)
    UserPermissionGroupMembership.objects.create(ffadminuser=user1, userpermissiongroup=user_permission_group_2)
    api_client.force_authenticate(user1)
    url = reverse('api-v1:organisations:organisation-groups-list', args=[organisation.id])
    response = api_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json['count'] == 1
    assert response_json['results'][0]['id'] == user_permission_group_1.id

def test_list_my_groups(organisation, api_client):
    if False:
        i = 10
        return i + 15
    user1 = FFAdminUser.objects.create(email='user1@example.com')
    user2 = FFAdminUser.objects.create(email='user2@example.com')
    user1.add_organisation(organisation)
    user2.add_organisation(organisation)
    user_permission_group_1 = UserPermissionGroup.objects.create(organisation=organisation, name='group1')
    UserPermissionGroupMembership.objects.create(ffadminuser=user1, userpermissiongroup=user_permission_group_1)
    user_permission_group_2 = UserPermissionGroup.objects.create(organisation=organisation, name='group2')
    UserPermissionGroupMembership.objects.create(ffadminuser=user2, userpermissiongroup=user_permission_group_2)
    api_client.force_authenticate(user1)
    url = reverse('api-v1:organisations:organisation-groups-my-groups', args=[organisation.id])
    response = api_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json['count'] == 1
    assert response_json['results'][0] == {'id': user_permission_group_1.id, 'name': user_permission_group_1.name}