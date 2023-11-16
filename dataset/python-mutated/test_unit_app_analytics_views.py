from datetime import date, timedelta
from app_analytics.dataclasses import UsageData
from app_analytics.views import SDKAnalyticsFlags
from django.urls import reverse
from rest_framework import status

def test_sdk_analytics_does_not_allow_bad_data(mocker, settings, environment):
    if False:
        return 10
    settings.INFLUXDB_TOKEN = 'some-token'
    data = {'bad': 'data'}
    request = mocker.MagicMock(data=data, environment=environment)
    view = SDKAnalyticsFlags(request=request)
    mocked_track_feature_eval = mocker.patch('app_analytics.views.track_feature_evaluation_influxdb')
    response = view.post(request)
    assert response.status_code == status.HTTP_200_OK
    mocked_track_feature_eval.assert_not_called()

def test_sdk_analytics_allows_valid_data(mocker, settings, environment, feature):
    if False:
        while True:
            i = 10
    settings.INFLUXDB_TOKEN = 'some-token'
    data = {feature.name: 12}
    request = mocker.MagicMock(data=data, environment=environment)
    view = SDKAnalyticsFlags(request=request)
    mocked_track_feature_eval = mocker.patch('app_analytics.views.track_feature_evaluation_influxdb')
    response = view.post(request)
    assert response.status_code == status.HTTP_200_OK
    mocked_track_feature_eval.assert_called_once_with(environment.id, data)

def test_get_usage_data(mocker, admin_client, organisation):
    if False:
        i = 10
        return i + 15
    url = reverse('api-v1:organisations:usage-data', args=[organisation.id])
    mocked_get_usage_data = mocker.patch('app_analytics.views.get_usage_data', autospec=True, return_value=[UsageData(flags=10, day=date.today()), UsageData(flags=10, day=date.today() - timedelta(days=1))])
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == [{'flags': 10, 'day': str(date.today()), 'identities': 0, 'traits': 0, 'environment_document': 0}, {'flags': 10, 'day': str(date.today() - timedelta(days=1)), 'identities': 0, 'traits': 0, 'environment_document': 0}]
    mocked_get_usage_data.assert_called_once_with(organisation)

def test_get_usage_data_for_non_admin_user_returns_403(mocker, test_user_client, organisation):
    if False:
        i = 10
        return i + 15
    url = reverse('api-v1:organisations:usage-data', args=[organisation.id])
    response = test_user_client.get(url)
    assert response.status_code == status.HTTP_403_FORBIDDEN

def test_get_total_usage_count(mocker, admin_client, organisation):
    if False:
        print('Hello World!')
    url = reverse('api-v1:organisations:usage-data-total-count', args=[organisation.id])
    mocked_get_total_events_count = mocker.patch('app_analytics.views.get_total_events_count', autospec=True, return_value=100)
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {'count': 100}
    mocked_get_total_events_count.assert_called_once_with(organisation)

def test_get_total_usage_count_for_non_admin_user_returns_403(mocker, test_user_client, organisation):
    if False:
        i = 10
        return i + 15
    url = reverse('api-v1:organisations:usage-data-total-count', args=[organisation.id])
    response = test_user_client.get(url)
    assert response.status_code == status.HTTP_403_FORBIDDEN