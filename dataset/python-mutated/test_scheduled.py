import abc
from abc import abstractmethod
from datetime import timedelta
from typing import Type
from unittest.mock import Mock
from django.db.models import QuerySet
from sentry.constants import ObjectStatus
from sentry.models.apiapplication import ApiApplication, ApiApplicationStatus
from sentry.models.repository import Repository
from sentry.models.scheduledeletion import BaseScheduledDeletion, RegionScheduledDeletion, ScheduledDeletion
from sentry.models.team import Team
from sentry.services.hybrid_cloud.user.service import user_service
from sentry.signals import pending_delete
from sentry.tasks.deletion.scheduled import reattempt_deletions, reattempt_deletions_control, run_scheduled_deletions, run_scheduled_deletions_control
from sentry.testutils.abstract import Abstract
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import control_silo_test, region_silo_test

class RegionalRunScheduleDeletionTest(abc.ABC, TestCase):
    __test__ = Abstract(__module__, __qualname__)

    @property
    @abstractmethod
    def ScheduledDeletion(self) -> Type[BaseScheduledDeletion]:
        if False:
            while True:
                i = 10
        raise NotImplementedError('Subclasses should implement')

    @abstractmethod
    def create_simple_deletion(self) -> QuerySet:
        if False:
            print('Hello World!')
        raise NotImplementedError('Subclasses should implement!')

    @abstractmethod
    def create_does_not_proceed_deletion(self) -> QuerySet:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Subclasses should implement!')

    @abstractmethod
    def run_scheduled_deletions(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Subclasses should implement')

    @abstractmethod
    def reattempt_deletions(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Subclasses should implement')

    def test_schedule_and_cancel(self):
        if False:
            while True:
                i = 10
        qs = self.create_simple_deletion()
        inst = qs.first()
        schedule = self.ScheduledDeletion.schedule(inst, days=0)
        self.ScheduledDeletion.cancel(inst)
        assert not self.ScheduledDeletion.objects.filter(id=schedule.id).exists()
        assert self.ScheduledDeletion.cancel(inst) is None

    def test_duplicate_schedule(self):
        if False:
            i = 10
            return i + 15
        qs = self.create_simple_deletion()
        inst = qs.first()
        first = self.ScheduledDeletion.schedule(inst, days=0)
        second = self.ScheduledDeletion.schedule(inst, days=1)
        assert first.id == second.id
        assert first.guid == second.guid
        assert second.date_scheduled - first.date_scheduled >= timedelta(days=1)

    def test_simple(self):
        if False:
            while True:
                i = 10
        qs = self.create_simple_deletion()
        inst = qs.first()
        schedule = self.ScheduledDeletion.schedule(instance=inst, days=0)
        with self.tasks():
            self.run_scheduled_deletions()
        assert not qs.exists()
        assert not self.ScheduledDeletion.objects.filter(id=schedule.id).exists()

    def test_should_proceed_check(self):
        if False:
            i = 10
            return i + 15
        qs = self.create_does_not_proceed_deletion()
        inst = qs.first()
        schedule = self.ScheduledDeletion.schedule(instance=inst, days=0)
        with self.tasks():
            self.run_scheduled_deletions()
        assert qs.exists()
        assert not self.ScheduledDeletion.objects.filter(id=schedule.id, in_progress=True).exists()

    def test_ignore_in_progress(self):
        if False:
            i = 10
            return i + 15
        qs = self.create_simple_deletion()
        inst = qs.first()
        schedule = self.ScheduledDeletion.schedule(instance=inst, days=0)
        schedule.update(in_progress=True)
        with self.tasks():
            self.run_scheduled_deletions()
        assert qs.exists()
        assert self.ScheduledDeletion.objects.filter(id=schedule.id, in_progress=True).exists()

    def test_future_schedule(self):
        if False:
            while True:
                i = 10
        qs = self.create_simple_deletion()
        inst = qs.first()
        schedule = self.ScheduledDeletion.schedule(instance=inst, days=1)
        with self.tasks():
            self.run_scheduled_deletions()
        assert qs.exists()
        assert self.ScheduledDeletion.objects.filter(id=schedule.id, in_progress=False).exists()

    def test_triggers_pending_delete_signal(self):
        if False:
            print('Hello World!')
        signal_handler = Mock()
        pending_delete.connect(signal_handler)
        qs = self.create_simple_deletion()
        inst = qs.first()
        self.ScheduledDeletion.schedule(instance=inst, actor=self.user, days=0)
        with self.tasks():
            self.run_scheduled_deletions()
        assert signal_handler.call_count == 1
        args = signal_handler.call_args_list[0][1]
        assert args['instance'] == inst
        assert args['actor'] == user_service.get_user(user_id=self.user.id)
        pending_delete.disconnect(signal_handler)

    def test_no_pending_delete_trigger_on_skipped_delete(self):
        if False:
            i = 10
            return i + 15
        qs = self.create_does_not_proceed_deletion()
        inst = qs.first()
        signal_handler = Mock()
        pending_delete.connect(signal_handler)
        self.ScheduledDeletion.schedule(instance=inst, actor=self.user, days=0)
        with self.tasks():
            self.run_scheduled_deletions()
        pending_delete.disconnect(signal_handler)
        assert signal_handler.call_count == 0

    def test_handle_missing_record(self):
        if False:
            print('Hello World!')
        qs = self.create_simple_deletion()
        inst = qs.first()
        assert inst is not None
        schedule = self.ScheduledDeletion.schedule(instance=inst, days=0)
        inst.delete()
        with self.tasks():
            self.run_scheduled_deletions()
        assert not self.ScheduledDeletion.objects.filter(id=schedule.id).exists()

    def test_reattempt_simple(self):
        if False:
            print('Hello World!')
        qs = self.create_simple_deletion()
        inst = qs.first()
        schedule = self.ScheduledDeletion.schedule(instance=inst, days=-3)
        schedule.update(in_progress=True)
        with self.tasks():
            self.reattempt_deletions()
        schedule.refresh_from_db()
        assert not schedule.in_progress

    def test_reattempt_ignore_recent_jobs(self):
        if False:
            while True:
                i = 10
        qs = self.create_simple_deletion()
        inst = qs.first()
        schedule = self.ScheduledDeletion.schedule(instance=inst, days=0)
        schedule.update(in_progress=True)
        with self.tasks():
            self.reattempt_deletions()
        schedule.refresh_from_db()
        assert schedule.in_progress is True

@region_silo_test(stable=True)
class RunRegionScheduledDeletionTest(RegionalRunScheduleDeletionTest):

    @property
    def ScheduledDeletion(self) -> Type[BaseScheduledDeletion]:
        if False:
            i = 10
            return i + 15
        return RegionScheduledDeletion

    def run_scheduled_deletions(self) -> None:
        if False:
            while True:
                i = 10
        return run_scheduled_deletions()

    def reattempt_deletions(self) -> None:
        if False:
            print('Hello World!')
        return reattempt_deletions()

    def create_simple_deletion(self) -> QuerySet:
        if False:
            while True:
                i = 10
        org = self.create_organization(name='test')
        team = self.create_team(organization=org, name='delete')
        return Team.objects.filter(id=team.id)

    def create_does_not_proceed_deletion(self) -> QuerySet:
        if False:
            print('Hello World!')
        org = self.create_organization(name='test')
        project = self.create_project(organization=org)
        repo = self.create_repo(project=project, name='example/example')
        assert repo.status == ObjectStatus.ACTIVE
        return Repository.objects.filter(id=repo.id)

@control_silo_test(stable=True)
class RunControlScheduledDeletionTest(RegionalRunScheduleDeletionTest):

    @property
    def ScheduledDeletion(self) -> Type[BaseScheduledDeletion]:
        if False:
            while True:
                i = 10
        return ScheduledDeletion

    def run_scheduled_deletions(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        return run_scheduled_deletions_control()

    def reattempt_deletions(self) -> None:
        if False:
            print('Hello World!')
        return reattempt_deletions_control()

    def create_simple_deletion(self) -> QuerySet:
        if False:
            return 10
        app = ApiApplication.objects.create(owner_id=self.user.id, allowed_origins='example.com')
        app.status = ApiApplicationStatus.pending_deletion
        app.save()
        return ApiApplication.objects.filter(id=app.id)

    def create_does_not_proceed_deletion(self) -> QuerySet:
        if False:
            print('Hello World!')
        app = ApiApplication.objects.create(owner_id=self.user.id, allowed_origins='example.com')
        app.status = ApiApplicationStatus.active
        app.save()
        return ApiApplication.objects.filter(id=app.id)