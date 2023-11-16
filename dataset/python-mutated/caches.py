from django.db.transaction import on_commit
from orgs.models import Organization
from orgs.tasks import refresh_org_cache_task
from orgs.utils import current_org, tmp_to_org
from common.cache import Cache, IntegerField
from common.utils import get_logger
from common.utils.timezone import local_zero_hour, local_monday
from users.models import UserGroup, User
from assets.models import Node, Domain, Asset
from accounts.models import Account
from terminal.models import Session
from perms.models import AssetPermission
logger = get_logger(__file__)

class OrgRelatedCache(Cache):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.current_org = Organization.get_instance(current_org.id)

    def get_current_org(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        暴露给子类控制组织的回调\n        1. 在交互式环境下能控制组织\n        2. 在 celery 任务下能控制组织\n        '
        return self.current_org

    def compute_values(self, *fields):
        if False:
            return 10
        with tmp_to_org(self.get_current_org()):
            return super().compute_values(*fields)

    def refresh_async(self, *fields):
        if False:
            i = 10
            return i + 15
        '\n        在事务提交之后再发送信号，防止因事务的隔离性导致未获得最新的数据\n        '

        def func():
            if False:
                i = 10
                return i + 15
            logger.debug(f'CACHE: Send refresh task {self}.{fields}')
            refresh_org_cache_task.delay(self, *fields)
        on_commit(func)

    def expire(self, *fields):
        if False:
            for i in range(10):
                print('nop')

        def func():
            if False:
                print('Hello World!')
            super(OrgRelatedCache, self).expire(*fields)
        on_commit(func)

class OrgResourceStatisticsCache(OrgRelatedCache):
    users_amount = IntegerField()
    assets_amount = IntegerField()
    new_users_amount_this_week = IntegerField()
    new_assets_amount_this_week = IntegerField()
    nodes_amount = IntegerField(queryset=Node.objects)
    domains_amount = IntegerField(queryset=Domain.objects)
    groups_amount = IntegerField(queryset=UserGroup.objects)
    accounts_amount = IntegerField(queryset=Account.objects)
    asset_perms_amount = IntegerField(queryset=AssetPermission.objects)
    total_count_online_users = IntegerField()
    total_count_online_sessions = IntegerField()
    total_count_today_active_assets = IntegerField()
    total_count_today_failed_sessions = IntegerField()

    def __init__(self, org):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.org = org

    def get_key_suffix(self):
        if False:
            while True:
                i = 10
        return f'org_{self.org.id}'

    def get_current_org(self):
        if False:
            i = 10
            return i + 15
        return self.org

    def get_users(self):
        if False:
            return 10
        return User.get_org_users(self.org)

    @staticmethod
    def get_assets():
        if False:
            while True:
                i = 10
        return Asset.objects.all()

    def compute_users_amount(self):
        if False:
            while True:
                i = 10
        users = self.get_users()
        return users.count()

    def compute_new_users_amount_this_week(self):
        if False:
            while True:
                i = 10
        monday_time = local_monday()
        users = self.get_users().filter(date_joined__gte=monday_time)
        return users.count()

    def compute_assets_amount(self):
        if False:
            i = 10
            return i + 15
        assets = self.get_assets()
        return assets.count()

    def compute_new_assets_amount_this_week(self):
        if False:
            i = 10
            return i + 15
        monday_time = local_monday()
        assets = self.get_assets().filter(date_created__gte=monday_time)
        return assets.count()

    @staticmethod
    def compute_total_count_online_users():
        if False:
            for i in range(10):
                print('nop')
        return Session.objects.filter(is_finished=False).values_list('user_id').distinct().count()

    @staticmethod
    def compute_total_count_online_sessions():
        if False:
            print('Hello World!')
        return Session.objects.filter(is_finished=False).count()

    @staticmethod
    def compute_total_count_today_active_assets():
        if False:
            i = 10
            return i + 15
        t = local_zero_hour()
        return Session.objects.filter(date_start__gte=t).values('asset_id').distinct().count()

    @staticmethod
    def compute_total_count_today_failed_sessions():
        if False:
            print('Hello World!')
        t = local_zero_hour()
        return Session.objects.filter(date_start__gte=t, is_success=False).count()