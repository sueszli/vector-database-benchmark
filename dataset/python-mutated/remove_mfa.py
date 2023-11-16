"""Custom management command to remove MFA for a user."""
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    """Remove MFA for a user."""

    def add_arguments(self, parser):
        if False:
            print('Hello World!')
        'Add the arguments.'
        parser.add_argument('mail', type=str)

    def handle(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Remove MFA for the supplied user (by mail).'
        mail = kwargs.get('mail')
        if not mail:
            raise KeyError('A mail is required')
        user = get_user_model()
        mfa_user = [*set(user.objects.filter(email=mail) | user.objects.filter(emailaddress__email=mail))]
        if len(mfa_user) == 0:
            print('No user with this mail associated')
        elif len(mfa_user) > 1:
            print('More than one user found with this mail')
        else:
            mfa_user[0].staticdevice_set.all().delete()
            mfa_user[0].totpdevice_set.all().delete()
            print(f'Removed all MFA methods for user {str(mfa_user[0])}')