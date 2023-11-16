import pytest
from django.utils.timezone import now
from awx.main.models import HostMetric

@pytest.mark.django_db
def test_host_metrics_generation():
    if False:
        i = 10
        return i + 15
    hostnames = [f'Host {i}' for i in range(100)]
    current_time = now()
    HostMetric.objects.bulk_create([HostMetric(hostname=h, last_automation=current_time) for h in hostnames])
    assert HostMetric.objects.count() == len(hostnames)
    assert sorted([s.hostname for s in HostMetric.objects.all()]) == sorted(hostnames)
    date_today = now().strftime('%Y-%m-%d')
    result = HostMetric.objects.filter(first_automation__startswith=date_today).count()
    assert result == len(hostnames)

@pytest.mark.django_db
def test_soft_delete():
    if False:
        return 10
    hostnames = [f'Host to delete {i}' for i in range(2)]
    current_time = now()
    HostMetric.objects.bulk_create([HostMetric(hostname=h, last_automation=current_time, automated_counter=42) for h in hostnames])
    hm = HostMetric.objects.get(hostname='Host to delete 0')
    assert hm.last_deleted is None
    last_deleted = None
    for _ in range(3):
        hm.soft_delete()
        if last_deleted is None:
            last_deleted = hm.last_deleted
        assert hm.deleted is True
        assert hm.deleted_counter == 1
        assert hm.last_deleted == last_deleted
        assert hm.automated_counter == 42
    hm = HostMetric.objects.get(hostname='Host to delete 1')
    assert hm.deleted is False
    assert hm.deleted_counter == 0
    assert hm.last_deleted is None
    assert hm.automated_counter == 42

@pytest.mark.django_db
def test_soft_restore():
    if False:
        print('Hello World!')
    current_time = now()
    HostMetric.objects.create(hostname='Host 1', last_automation=current_time, deleted=True)
    HostMetric.objects.create(hostname='Host 2', last_automation=current_time, deleted=True, last_deleted=current_time)
    HostMetric.objects.create(hostname='Host 3', last_automation=current_time, deleted=False, last_deleted=current_time)
    HostMetric.objects.all().update(automated_counter=42, deleted_counter=10)
    for hm in HostMetric.objects.all():
        for _ in range(3):
            hm.soft_restore()
            assert hm.deleted is False
            assert hm.automated_counter == 42 and hm.deleted_counter == 10
            if hm.hostname == 'Host 1':
                assert hm.last_deleted is None
            else:
                assert hm.last_deleted == current_time