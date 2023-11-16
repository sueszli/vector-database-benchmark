import asyncio
from unittest import mock
import pytest
from prefect.server.database.alembic_commands import alembic_downgrade, alembic_revision, alembic_stamp, alembic_upgrade
from prefect.utilities.asyncutils import run_sync_in_worker_thread

class TestAlembicCommands:

    @mock.patch('alembic.command.upgrade')
    def test_alembic_upgrade_defaults(self, mocked):
        if False:
            return 10
        alembic_upgrade()
        (args, kwargs) = mocked.call_args
        assert mocked.call_count == 1
        assert args[1] == 'head'
        assert kwargs['sql'] is False

    @mock.patch('alembic.command.upgrade')
    def test_alembic_upgrade_passed_params(self, mocked):
        if False:
            for i in range(10):
                print('nop')
        alembic_upgrade('revision123', dry_run=True)
        (args, kwargs) = mocked.call_args
        assert mocked.call_count == 1
        assert args[1] == 'revision123'
        assert kwargs['sql'] is True

    @mock.patch('alembic.command.downgrade')
    def test_alembic_downgrade_defaults(self, mocked):
        if False:
            return 10
        alembic_downgrade()
        (args, kwargs) = mocked.call_args
        assert mocked.call_count == 1
        assert args[1] == 'base'
        assert kwargs['sql'] is False

    @mock.patch('alembic.command.downgrade')
    def test_alembic_downgrade_passed_params(self, mocked):
        if False:
            print('Hello World!')
        alembic_downgrade('revision123', dry_run=True)
        (args, kwargs) = mocked.call_args
        assert mocked.call_count == 1
        assert args[1] == 'revision123'
        assert kwargs['sql'] is True

    @mock.patch('alembic.command.revision')
    def test_alembic_revision_defaults(self, mocked):
        if False:
            return 10
        alembic_revision()
        (_, kwargs) = mocked.call_args
        assert mocked.call_count == 1
        assert kwargs['message'] is None
        assert kwargs['autogenerate'] is False

    @mock.patch('alembic.command.revision')
    def test_alembic_revision_passed_params(self, mocked):
        if False:
            i = 10
            return i + 15
        alembic_revision(message='new_revision', autogenerate=True)
        (_, kwargs) = mocked.call_args
        assert mocked.call_count == 1
        assert kwargs['message'] == 'new_revision'
        assert kwargs['autogenerate'] is True

    @mock.patch('alembic.command.stamp')
    def test_alembic_stamp(self, mocked):
        if False:
            for i in range(10):
                print('nop')
        alembic_stamp(revision='abcdef')
        (_, kwargs) = mocked.call_args
        assert mocked.call_count == 1
        assert kwargs['revision'] == 'abcdef'

    async def test_concurrent_upgrade(self):
        jobs = [run_sync_in_worker_thread(alembic_upgrade) for _ in range(0, 10)]
        await asyncio.gather(*jobs)

    @pytest.mark.skip(reason="This test is occasionally failing on CI because the tables aren't being restored after the downgrade, which makes the DB cleanup fixture error for the rest of the test suite")
    async def test_concurrent_downgrade_upgrade(self):
        try:
            jobs = []
            for _ in range(0, 2):
                jobs.append(run_sync_in_worker_thread(alembic_downgrade))
                jobs.append(run_sync_in_worker_thread(alembic_upgrade))
            await asyncio.gather(*jobs)
        finally:
            await run_sync_in_worker_thread(alembic_upgrade)