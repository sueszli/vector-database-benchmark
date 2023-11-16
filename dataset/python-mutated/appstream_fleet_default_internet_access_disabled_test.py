from unittest import mock
from prowler.providers.aws.services.appstream.appstream_service import Fleet
AWS_REGION = 'eu-west-1'

class Test_appstream_fleet_default_internet_access_disabled:

    def test_no_fleets(self):
        if False:
            return 10
        appstream_client = mock.MagicMock
        appstream_client.fleets = []
        with mock.patch('prowler.providers.aws.services.appstream.appstream_service.AppStream', new=appstream_client):
            from prowler.providers.aws.services.appstream.appstream_fleet_default_internet_access_disabled.appstream_fleet_default_internet_access_disabled import appstream_fleet_default_internet_access_disabled
            check = appstream_fleet_default_internet_access_disabled()
            result = check.execute()
            assert len(result) == 0

    def test_one_fleet_internet_access_enabled(self):
        if False:
            print('Hello World!')
        appstream_client = mock.MagicMock
        appstream_client.fleets = []
        fleet1 = Fleet(arn='arn', name='test-fleet', max_user_duration_in_seconds=900, disconnect_timeout_in_seconds=900, idle_disconnect_timeout_in_seconds=900, enable_default_internet_access=True, region=AWS_REGION)
        appstream_client.fleets.append(fleet1)
        with mock.patch('prowler.providers.aws.services.appstream.appstream_service.AppStream', new=appstream_client):
            from prowler.providers.aws.services.appstream.appstream_fleet_default_internet_access_disabled.appstream_fleet_default_internet_access_disabled import appstream_fleet_default_internet_access_disabled
            check = appstream_fleet_default_internet_access_disabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].resource_arn == fleet1.arn
            assert result[0].region == fleet1.region
            assert result[0].resource_id == fleet1.name
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'Fleet {fleet1.name} has default internet access enabled.'
            assert result[0].resource_tags == []

    def test_one_fleet_internet_access_disbaled(self):
        if False:
            return 10
        appstream_client = mock.MagicMock
        appstream_client.fleets = []
        fleet1 = Fleet(arn='arn', name='test-fleet', max_user_duration_in_seconds=900, disconnect_timeout_in_seconds=900, idle_disconnect_timeout_in_seconds=900, enable_default_internet_access=False, region=AWS_REGION)
        appstream_client.fleets.append(fleet1)
        with mock.patch('prowler.providers.aws.services.appstream.appstream_service.AppStream', new=appstream_client):
            from prowler.providers.aws.services.appstream.appstream_fleet_default_internet_access_disabled.appstream_fleet_default_internet_access_disabled import appstream_fleet_default_internet_access_disabled
            check = appstream_fleet_default_internet_access_disabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].resource_arn == fleet1.arn
            assert result[0].region == fleet1.region
            assert result[0].resource_id == fleet1.name
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'Fleet {fleet1.name} has default internet access disabled.'
            assert result[0].resource_tags == []

    def test_two_fleets_internet_access_one_enabled_two_disabled(self):
        if False:
            i = 10
            return i + 15
        appstream_client = mock.MagicMock
        appstream_client.fleets = []
        fleet1 = Fleet(arn='arn', name='test-fleet-1', max_user_duration_in_seconds=900, disconnect_timeout_in_seconds=900, idle_disconnect_timeout_in_seconds=900, enable_default_internet_access=True, region=AWS_REGION)
        fleet2 = Fleet(arn='arn', name='test-fleet-2', max_user_duration_in_seconds=900, disconnect_timeout_in_seconds=900, idle_disconnect_timeout_in_seconds=900, enable_default_internet_access=False, region=AWS_REGION)
        appstream_client.fleets.append(fleet1)
        appstream_client.fleets.append(fleet2)
        with mock.patch('prowler.providers.aws.services.appstream.appstream_service.AppStream', new=appstream_client):
            from prowler.providers.aws.services.appstream.appstream_fleet_default_internet_access_disabled.appstream_fleet_default_internet_access_disabled import appstream_fleet_default_internet_access_disabled
            check = appstream_fleet_default_internet_access_disabled()
            result = check.execute()
            assert len(result) == 2
            for res in result:
                if res.resource_id == fleet1.name:
                    assert result[0].resource_arn == fleet1.arn
                    assert result[0].region == fleet1.region
                    assert result[0].resource_id == fleet1.name
                    assert result[0].status == 'FAIL'
                    assert result[0].status_extended == f'Fleet {fleet1.name} has default internet access enabled.'
                    assert result[0].resource_tags == []
                if res.resource_id == fleet2.name:
                    assert result[1].resource_arn == fleet2.arn
                    assert result[1].region == fleet2.region
                    assert result[1].resource_id == fleet2.name
                    assert result[1].status == 'PASS'
                    assert result[1].status_extended == f'Fleet {fleet2.name} has default internet access disabled.'
                    assert result[1].resource_tags == []