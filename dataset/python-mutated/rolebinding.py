from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q
from django.db.models.signals import post_save
from django.utils.translation import gettext_lazy as _
from rest_framework.serializers import ValidationError
from common.db.models import JMSBaseModel, CASCADE_SIGNAL_SKIP
from common.utils import lazyproperty
from orgs.utils import current_org, tmp_to_root_org
from .role import Role
from ..const import Scope
__all__ = ['RoleBinding', 'SystemRoleBinding', 'OrgRoleBinding']

class RoleBindingManager(models.Manager):

    def bulk_create(self, objs, batch_size=None, ignore_conflicts=False):
        if False:
            return 10
        objs = super().bulk_create(objs, batch_size=batch_size, ignore_conflicts=ignore_conflicts)
        for i in objs:
            post_save.send(i.__class__, instance=i, created=True)
        return objs

    def get_queryset(self):
        if False:
            for i in range(10):
                print('nop')
        queryset = super(RoleBindingManager, self).get_queryset()
        q = Q(scope=Scope.system, org__isnull=True)
        if not current_org.is_root():
            q |= Q(org_id=current_org.id, scope=Scope.org)
        queryset = queryset.filter(q)
        return queryset

    def root_all(self):
        if False:
            for i in range(10):
                print('nop')
        queryset = super().get_queryset()
        if current_org.is_root():
            return queryset
        return self.get_queryset()

class RoleBinding(JMSBaseModel):
    Scope = Scope
    ' 定义 用户-角色 关系 '
    scope = models.CharField(max_length=128, choices=Scope.choices, default=Scope.system, verbose_name=_('Scope'))
    user = models.ForeignKey('users.User', related_name='role_bindings', on_delete=CASCADE_SIGNAL_SKIP, verbose_name=_('User'))
    role = models.ForeignKey(Role, related_name='role_bindings', on_delete=models.CASCADE, verbose_name=_('Role'))
    org = models.ForeignKey('orgs.Organization', related_name='role_bindings', blank=True, null=True, on_delete=models.CASCADE, verbose_name=_('Organization'))
    objects = RoleBindingManager()

    class Meta:
        verbose_name = _('Role binding')
        unique_together = [('user', 'role', 'org')]

    def __str__(self):
        if False:
            return 10
        display = '{role} -> {user}'.format(user=self.user, role=self.role)
        if self.org:
            display += ' | {org}'.format(org=self.org)
        return display

    @property
    def org_name(self):
        if False:
            for i in range(10):
                print('nop')
        if self.org:
            return self.org.name
        return ''

    def save(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.scope = self.role.scope
        self.clean()
        return super().save(*args, **kwargs)

    @classmethod
    def get_user_perms(cls, user):
        if False:
            print('Hello World!')
        roles = cls.get_user_roles(user)
        return Role.get_roles_perms(roles)

    @classmethod
    def get_role_users(cls, role):
        if False:
            for i in range(10):
                print('nop')
        from users.models import User
        bindings = cls.objects.root_all().filter(role=role, scope=role.scope)
        user_ids = bindings.values_list('user', flat=True).distinct()
        return User.objects.filter(id__in=user_ids)

    @classmethod
    def get_user_roles(cls, user):
        if False:
            while True:
                i = 10
        bindings = cls.objects.filter(user=user)
        roles_id = bindings.values_list('role', flat=True).distinct()
        return Role.objects.filter(id__in=roles_id)

    @lazyproperty
    def user_display(self):
        if False:
            i = 10
            return i + 15
        return self.user.name

    @lazyproperty
    def role_display(self):
        if False:
            return 10
        return self.role.display_name

    def is_scope_org(self):
        if False:
            while True:
                i = 10
        return self.scope == Scope.org

    @staticmethod
    def orgs_order_by_name(orgs):
        if False:
            i = 10
            return i + 15
        from orgs.models import Organization
        default_system_org_ids = [Organization.DEFAULT_ID, Organization.SYSTEM_ID]
        default_system_orgs = orgs.filter(id__in=default_system_org_ids)
        return default_system_orgs | orgs.exclude(id__in=default_system_org_ids).order_by('name')

    @classmethod
    def get_user_joined_orgs(cls, user):
        if False:
            for i in range(10):
                print('nop')
        from orgs.models import Organization
        org_ids = cls.objects.filter(user=user, scope=Scope.org).values_list('org', flat=True).distinct()
        return Organization.objects.filter(id__in=org_ids)

    @classmethod
    def get_user_has_the_perm_orgs(cls, perm, user):
        if False:
            while True:
                i = 10
        from orgs.models import Organization
        roles = Role.get_roles_by_perm(perm)
        with tmp_to_root_org():
            bindings = list(cls.objects.root_all().filter(role__in=roles, user=user))
        system_bindings = [b for b in bindings if b.scope == Role.Scope.system.value]
        if perm == 'rbac.view_workbench':
            all_orgs = user.orgs.all().distinct()
        else:
            all_orgs = Organization.objects.all()
        if not settings.XPACK_ENABLED:
            all_orgs = all_orgs.filter(id=Organization.DEFAULT_ID)
        if system_bindings:
            orgs = all_orgs
        else:
            org_ids = [b.org.id for b in bindings if b.org]
            orgs = all_orgs.filter(id__in=org_ids)
        orgs = cls.orgs_order_by_name(orgs)
        workbench_perm = 'rbac.view_workbench'
        if orgs and perm != workbench_perm and user.has_perm('orgs.view_rootorg'):
            root_org = Organization.root()
            orgs = [root_org, *list(orgs)]
        elif orgs and perm == workbench_perm and user.has_perm('orgs.view_alljoinedorg'):
            root_org = Organization.root()
            root_org.name = _('All organizations')
            orgs = [root_org, *list(orgs)]
        return orgs

class OrgRoleBindingManager(RoleBindingManager):

    def get_queryset(self):
        if False:
            return 10
        queryset = super(RoleBindingManager, self).get_queryset()
        if current_org.is_root():
            queryset = queryset.none()
        else:
            queryset = queryset.filter(org_id=current_org.id, scope=Scope.org)
        return queryset

class OrgRoleBinding(RoleBinding):
    objects = OrgRoleBindingManager()

    def save(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.org_id = current_org.id
        self.scope = Scope.org
        return super().save(*args, **kwargs)

    def delete(self, **kwargs):
        if False:
            return 10
        has_other_role = self.__class__.objects.filter(user=self.user, scope=self.scope).exclude(id=self.id).exists()
        if not has_other_role:
            error = _('User last role in org, can not be delete, you can remove user from org instead')
            raise ValidationError({'error': error})
        return super().delete(**kwargs)

    class Meta:
        proxy = True
        verbose_name = _('Organization role binding')

class SystemRoleBindingManager(RoleBindingManager):

    def get_queryset(self):
        if False:
            while True:
                i = 10
        queryset = super(RoleBindingManager, self).get_queryset().filter(scope=Scope.system)
        return queryset

class SystemRoleBinding(RoleBinding):
    objects = SystemRoleBindingManager()

    class Meta:
        proxy = True
        verbose_name = _('System role binding')

    def save(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.scope = Scope.system
        return super().save(*args, **kwargs)

    def clean(self):
        if False:
            print('Hello World!')
        kwargs = dict(role=self.role, user=self.user, scope=self.scope)
        exists = self.__class__.objects.filter(**kwargs).exists()
        if exists:
            msg = "Duplicate for key 'role_user' of system role binding, {}_{}".format(self.role.id, self.user.id)
            raise ValidationError(msg)