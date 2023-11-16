from django.apps import apps
from django.conf import settings
from django.contrib.auth.models import Group, UserManager
from django.contrib.sites.models import Site
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils.encoding import force_str
from django.utils.translation import gettext_lazy as _
from cms.models import Page
from cms.models.managers import GlobalPagePermissionManager, PagePermissionManager
(user_app_name, user_model_name) = settings.AUTH_USER_MODEL.rsplit('.', 1)
User = None
try:
    User = apps.get_registered_model(user_app_name, user_model_name)
except KeyError:
    pass
if User is None:
    raise ImproperlyConfigured('You have defined a custom user model %s, but the app %s is not in settings.INSTALLED_APPS' % (settings.AUTH_USER_MODEL, user_app_name))
ACCESS_PAGE = 1
ACCESS_CHILDREN = 2
ACCESS_PAGE_AND_CHILDREN = 3
ACCESS_DESCENDANTS = 4
ACCESS_PAGE_AND_DESCENDANTS = 5
MASK_PAGE = 1
MASK_CHILDREN = 2
MASK_DESCENDANTS = 4
ACCESS_CHOICES = ((ACCESS_PAGE, _('Current page')), (ACCESS_CHILDREN, _('Page children (immediate)')), (ACCESS_PAGE_AND_CHILDREN, _('Page and children (immediate)')), (ACCESS_DESCENDANTS, _('Page descendants')), (ACCESS_PAGE_AND_DESCENDANTS, _('Page and descendants')))

class AbstractPagePermission(models.Model):
    """Abstract page permissions
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name=_('user'), blank=True, null=True)
    group = models.ForeignKey(Group, on_delete=models.CASCADE, verbose_name=_('group'), blank=True, null=True)
    can_change = models.BooleanField(_('can edit'), default=True)
    can_add = models.BooleanField(_('can add'), default=True)
    can_delete = models.BooleanField(_('can delete'), default=True)
    can_change_advanced_settings = models.BooleanField(_('can change advanced settings'), default=False)
    can_publish = models.BooleanField(_('can publish'), default=True)
    can_change_permissions = models.BooleanField(_('can change permissions'), default=False, help_text=_('on page level'))
    can_move_page = models.BooleanField(_('can move'), default=True)
    can_view = models.BooleanField(_('view restricted'), default=False, help_text=_('frontend view restriction'))

    class Meta:
        abstract = True
        app_label = 'cms'

    def clean(self):
        if False:
            while True:
                i = 10
        super().clean()
        if not self.user and (not self.group):
            raise ValidationError(_('Please select user or group.'))
        if self.can_change:
            return
        if self.can_add:
            message = _("Users can't create a page without permissions to change the created page. Edit permissions required.")
            raise ValidationError(message)
        if self.can_delete:
            message = _("Users can't delete a page without permissions to change the page. Edit permissions required.")
            raise ValidationError(message)
        if self.can_publish:
            message = _("Users can't publish a page without permissions to change the page. Edit permissions required.")
            raise ValidationError(message)
        if self.can_change_advanced_settings:
            message = _("Users can't change page advanced settings without permissions to change the page. Edit permissions required.")
            raise ValidationError(message)
        if self.can_change_permissions:
            message = _("Users can't change page permissions without permissions to change the page. Edit permissions required.")
            raise ValidationError(message)
        if self.can_move_page:
            message = _("Users can't move a page without permissions to change the page. Edit permissions required.")
            raise ValidationError(message)

    @property
    def audience(self):
        if False:
            return 10
        'Return audience by priority, so: All or User, Group\n        '
        targets = filter(lambda item: item, (self.user, self.group))
        return ', '.join([force_str(t) for t in targets]) or 'No one'

    def save(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not self.user and (not self.group):
            return
        return super().save(*args, **kwargs)

    def get_configured_actions(self):
        if False:
            print('Hello World!')
        actions = [action for action in self.get_permissions_by_action() if self.has_configured_action(action)]
        return actions

    def has_configured_action(self, action):
        if False:
            for i in range(10):
                print('nop')
        permissions = self.get_permissions_by_action()[action]
        return all((getattr(self, perm) for perm in permissions))

    @classmethod
    def get_all_permissions(cls):
        if False:
            for i in range(10):
                print('nop')
        perms = ['can_add', 'can_change', 'can_delete', 'can_publish', 'can_change_advanced_settings', 'can_change_permissions', 'can_move_page', 'can_view']
        return perms

    @classmethod
    def get_permissions_by_action(cls):
        if False:
            return 10
        permissions_by_action = {'add_page': ['can_add', 'can_change'], 'change_page': ['can_change'], 'change_page_advanced_settings': ['can_change', 'can_change_advanced_settings'], 'change_page_permissions': ['can_change', 'can_change_permissions'], 'delete_page': ['can_change', 'can_delete'], 'delete_page_translation': ['can_change', 'can_delete'], 'move_page': ['can_change', 'can_move_page'], 'publish_page': ['can_change', 'can_publish'], 'view_page': ['can_view']}
        return permissions_by_action

class GlobalPagePermission(AbstractPagePermission):
    """Permissions for all pages (global).
    """
    can_recover_page = models.BooleanField(verbose_name=_('can recover pages'), default=True, help_text=_('can recover any deleted page'))
    sites = models.ManyToManyField(to=Site, blank=True, help_text=_('If none selected, user haves granted permissions to all sites.'), verbose_name=_('sites'))
    objects = GlobalPagePermissionManager()

    class Meta:
        verbose_name = _('Page global permission')
        verbose_name_plural = _('Pages global permissions')
        app_label = 'cms'

    def __str__(self):
        if False:
            return 10
        return '%s :: GLOBAL' % self.audience

class PagePermission(AbstractPagePermission):
    """Page permissions for single page
    """
    grant_on = models.IntegerField(_('Grant on'), choices=ACCESS_CHOICES, default=ACCESS_PAGE_AND_DESCENDANTS)
    page = models.ForeignKey(Page, on_delete=models.CASCADE, null=True, blank=True, verbose_name=_('page'))
    objects = PagePermissionManager()

    class Meta:
        verbose_name = _('Page permission')
        verbose_name_plural = _('Page permissions')
        app_label = 'cms'

    def __str__(self):
        if False:
            while True:
                i = 10
        page = self.page_id and force_str(self.page) or 'None'
        return f'{page} :: {self.audience} has: {force_str(self.get_grant_on_display())}'

    def clean(self):
        if False:
            print('Hello World!')
        super().clean()
        if self.can_add and self.grant_on == ACCESS_PAGE:
            message = _("Add page permission requires also access to children, or descendants, otherwise added page can't be changed by its creator.")
            raise ValidationError(message)

    def get_page_ids(self):
        if False:
            while True:
                i = 10
        if self.grant_on & MASK_PAGE:
            yield self.page_id
        if self.grant_on & MASK_CHILDREN:
            children = self.page.get_child_pages().values_list('pk', flat=True)
            yield from children
        elif self.grant_on & MASK_DESCENDANTS:
            node = self.page.node
            if node._has_cached_hierarchy():
                descendants = (node.item.pk for node in node.get_cached_descendants())
            else:
                descendants = self.page.get_descendant_pages().values_list('pk', flat=True).iterator()
            yield from descendants

class PageUserManager(UserManager):
    use_in_migrations = False

class PageUser(User):
    """Cms specific user data, required for permission system
    """
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='created_users')
    objects = PageUserManager()

    class Meta:
        verbose_name = _('User (page)')
        verbose_name_plural = _('Users (page)')
        app_label = 'cms'

class PageUserGroup(Group):
    """Cms specific group data, required for permission system
    """
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='created_usergroups')

    class Meta:
        verbose_name = _('User group (page)')
        verbose_name_plural = _('User groups (page)')
        app_label = 'cms'