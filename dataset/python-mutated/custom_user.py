from django.contrib.auth.models import AbstractBaseUser, AbstractUser, BaseUserManager, Group, Permission, PermissionsMixin, UserManager
from django.db import models

class CustomUserManager(BaseUserManager):

    def create_user(self, email, date_of_birth, password=None, **fields):
        if False:
            print('Hello World!')
        '\n        Creates and saves a User with the given email and password.\n        '
        if not email:
            raise ValueError('Users must have an email address')
        user = self.model(email=self.normalize_email(email), date_of_birth=date_of_birth, **fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password, date_of_birth, **fields):
        if False:
            i = 10
            return i + 15
        u = self.create_user(email, password=password, date_of_birth=date_of_birth, **fields)
        u.is_admin = True
        u.save(using=self._db)
        return u

class CustomUser(AbstractBaseUser):
    email = models.EmailField(verbose_name='email address', max_length=255, unique=True)
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    date_of_birth = models.DateField()
    first_name = models.CharField(max_length=50)
    custom_objects = CustomUserManager()
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['date_of_birth', 'first_name']

    def __str__(self):
        if False:
            print('Hello World!')
        return self.email

    def get_group_permissions(self, obj=None):
        if False:
            i = 10
            return i + 15
        return set()

    def get_all_permissions(self, obj=None):
        if False:
            while True:
                i = 10
        return set()

    def has_perm(self, perm, obj=None):
        if False:
            for i in range(10):
                print('nop')
        return True

    def has_perms(self, perm_list, obj=None):
        if False:
            return 10
        return True

    def has_module_perms(self, app_label):
        if False:
            return 10
        return True

    @property
    def is_staff(self):
        if False:
            return 10
        return self.is_admin

class RemoveGroupsAndPermissions:
    """
    A context manager to temporarily remove the groups and user_permissions M2M
    fields from the AbstractUser class, so they don't clash with the
    related_name sets.
    """

    def __enter__(self):
        if False:
            print('Hello World!')
        self._old_au_local_m2m = AbstractUser._meta.local_many_to_many
        self._old_pm_local_m2m = PermissionsMixin._meta.local_many_to_many
        groups = models.ManyToManyField(Group, blank=True)
        groups.contribute_to_class(PermissionsMixin, 'groups')
        user_permissions = models.ManyToManyField(Permission, blank=True)
        user_permissions.contribute_to_class(PermissionsMixin, 'user_permissions')
        PermissionsMixin._meta.local_many_to_many = [groups, user_permissions]
        AbstractUser._meta.local_many_to_many = [groups, user_permissions]

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            while True:
                i = 10
        AbstractUser._meta.local_many_to_many = self._old_au_local_m2m
        PermissionsMixin._meta.local_many_to_many = self._old_pm_local_m2m

class CustomUserWithoutIsActiveField(AbstractBaseUser):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    objects = UserManager()
    USERNAME_FIELD = 'username'
with RemoveGroupsAndPermissions():

    class ExtensionUser(AbstractUser):
        date_of_birth = models.DateField()
        custom_objects = UserManager()
        REQUIRED_FIELDS = AbstractUser.REQUIRED_FIELDS + ['date_of_birth']