from django.db import models
from django.db.models import Q
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from netbox.models import NestedGroupModel, PrimaryModel
from netbox.models.features import ContactsMixin
__all__ = ('Tenant', 'TenantGroup')

class TenantGroup(NestedGroupModel):
    """
    An arbitrary collection of Tenants.
    """
    name = models.CharField(verbose_name=_('name'), max_length=100, unique=True)
    slug = models.SlugField(verbose_name=_('slug'), max_length=100, unique=True)

    class Meta:
        ordering = ['name']
        verbose_name = _('tenant group')
        verbose_name_plural = _('tenant groups')

    def get_absolute_url(self):
        if False:
            i = 10
            return i + 15
        return reverse('tenancy:tenantgroup', args=[self.pk])

class Tenant(ContactsMixin, PrimaryModel):
    """
    A Tenant represents an organization served by the NetBox owner. This is typically a customer or an internal
    department.
    """
    name = models.CharField(verbose_name=_('name'), max_length=100)
    slug = models.SlugField(verbose_name=_('slug'), max_length=100)
    group = models.ForeignKey(to='tenancy.TenantGroup', on_delete=models.SET_NULL, related_name='tenants', blank=True, null=True)
    clone_fields = ('group', 'description')

    class Meta:
        ordering = ['name']
        constraints = (models.UniqueConstraint(fields=('group', 'name'), name='%(app_label)s_%(class)s_unique_group_name', violation_error_message=_('Tenant name must be unique per group.')), models.UniqueConstraint(fields=('name',), name='%(app_label)s_%(class)s_unique_name', condition=Q(group__isnull=True)), models.UniqueConstraint(fields=('group', 'slug'), name='%(app_label)s_%(class)s_unique_group_slug', violation_error_message=_('Tenant slug must be unique per group.')), models.UniqueConstraint(fields=('slug',), name='%(app_label)s_%(class)s_unique_slug', condition=Q(group__isnull=True)))
        verbose_name = _('tenant')
        verbose_name_plural = _('tenants')

    def __str__(self):
        if False:
            print('Hello World!')
        return self.name

    def get_absolute_url(self):
        if False:
            for i in range(10):
                print('nop')
        return reverse('tenancy:tenant', args=[self.pk])