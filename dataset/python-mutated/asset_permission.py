import logging
from django.db import models
from django.db.models import Q
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from accounts.const import AliasAccount
from accounts.models import Account
from assets.models import Asset
from common.utils import date_expired_default
from common.utils.timezone import local_now
from orgs.mixins.models import JMSOrgBaseModel
from orgs.mixins.models import OrgManager
from perms.const import ActionChoices
from users.models import User
__all__ = ['AssetPermission', 'ActionChoices', 'AssetPermissionQuerySet']
logger = logging.getLogger('jumpserver.permissions')

class AssetPermissionQuerySet(models.QuerySet):

    def active(self):
        if False:
            while True:
                i = 10
        return self.filter(is_active=True)

    def valid(self):
        if False:
            print('Hello World!')
        return self.active().filter(date_start__lt=timezone.now()).filter(date_expired__gt=timezone.now())

    def inactive(self):
        if False:
            return 10
        return self.filter(is_active=False)

    def invalid(self):
        if False:
            i = 10
            return i + 15
        now = timezone.now()
        q = Q(is_active=False) | Q(date_start__gt=now) | Q(date_expired__lt=now)
        return self.filter(q)

    def filter_by_accounts(self, accounts):
        if False:
            while True:
                i = 10
        q = Q(accounts__contains=list(accounts)) | Q(accounts__contains=AliasAccount.ALL.value)
        return self.filter(q)

class AssetPermissionManager(OrgManager):

    def valid(self):
        if False:
            i = 10
            return i + 15
        return self.get_queryset().valid()

    def get_expired_permissions(self):
        if False:
            while True:
                i = 10
        now = local_now()
        return self.get_queryset().filter(Q(date_start__lte=now) | Q(date_expired__gte=now))

def default_protocols():
    if False:
        print('Hello World!')
    return ['all']

class AssetPermission(JMSOrgBaseModel):
    name = models.CharField(max_length=128, verbose_name=_('Name'))
    users = models.ManyToManyField('users.User', related_name='%(class)ss', blank=True, verbose_name=_('User'))
    user_groups = models.ManyToManyField('users.UserGroup', related_name='%(class)ss', blank=True, verbose_name=_('User group'))
    assets = models.ManyToManyField('assets.Asset', related_name='granted_by_permissions', blank=True, verbose_name=_('Asset'))
    nodes = models.ManyToManyField('assets.Node', related_name='granted_by_permissions', blank=True, verbose_name=_('Nodes'))
    accounts = models.JSONField(default=list, verbose_name=_('Account'))
    protocols = models.JSONField(default=default_protocols, verbose_name=_('Protocols'))
    actions = models.IntegerField(default=ActionChoices.connect, verbose_name=_('Actions'))
    date_start = models.DateTimeField(default=timezone.now, db_index=True, verbose_name=_('Date start'))
    date_expired = models.DateTimeField(default=date_expired_default, db_index=True, verbose_name=_('Date expired'))
    is_active = models.BooleanField(default=True, verbose_name=_('Active'))
    from_ticket = models.BooleanField(default=False, verbose_name=_('From ticket'))
    objects = AssetPermissionManager.from_queryset(AssetPermissionQuerySet)()

    class Meta:
        unique_together = [('org_id', 'name')]
        verbose_name = _('Asset permission')
        ordering = ('name',)
        permissions = []

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.name

    @property
    def is_expired(self):
        if False:
            return 10
        if self.date_expired > timezone.now() > self.date_start:
            return False
        return True

    @property
    def is_valid(self):
        if False:
            i = 10
            return i + 15
        if not self.is_expired and self.is_active:
            return True
        return False

    def get_all_users(self):
        if False:
            for i in range(10):
                print('nop')
        from users.models import User
        user_ids = self.users.all().values_list('id', flat=True)
        group_ids = self.user_groups.all().values_list('id', flat=True)
        user_ids = list(user_ids)
        group_ids = list(group_ids)
        qs1_ids = User.objects.filter(id__in=user_ids).distinct().values_list('id', flat=True)
        qs2_ids = User.objects.filter(groups__id__in=group_ids).distinct().values_list('id', flat=True)
        qs_ids = list(qs1_ids) + list(qs2_ids)
        qs = User.objects.filter(id__in=qs_ids)
        return qs

    def get_all_assets(self, flat=False):
        if False:
            while True:
                i = 10
        from assets.models import Node
        nodes_keys = self.nodes.all().values_list('key', flat=True)
        asset_ids = set(self.assets.all().values_list('id', flat=True))
        nodes_asset_ids = Node.get_nodes_all_asset_ids_by_keys(nodes_keys)
        asset_ids.update(nodes_asset_ids)
        if flat:
            return asset_ids
        assets = Asset.objects.filter(id__in=asset_ids)
        return assets

    def get_all_accounts(self, flat=False):
        if False:
            for i in range(10):
                print('nop')
        '\n         :return: 返回授权的所有账号对象 Account\n        '
        asset_ids = self.get_all_assets(flat=True)
        q = Q(asset_id__in=asset_ids)
        if AliasAccount.ALL not in self.accounts:
            q &= Q(username__in=self.accounts)
        accounts = Account.objects.filter(q).order_by('asset__name', 'name', 'username')
        if not flat:
            return accounts
        return accounts.values_list('id', flat=True)

    @classmethod
    def get_all_users_for_perms(cls, perm_ids, flat=False):
        if False:
            print('Hello World!')
        user_ids = cls.users.through.objects.filter(assetpermission_id__in=perm_ids).values_list('user_id', flat=True).distinct()
        group_ids = cls.user_groups.through.objects.filter(assetpermission_id__in=perm_ids).values_list('usergroup_id', flat=True).distinct()
        group_user_ids = User.groups.through.objects.filter(usergroup_id__in=group_ids).values_list('user_id', flat=True).distinct()
        user_ids = set(user_ids) | set(group_user_ids)
        if flat:
            return user_ids
        return User.objects.filter(id__in=user_ids)