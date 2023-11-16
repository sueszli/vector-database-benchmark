import logging
from argparse import ArgumentParser
from typing import Any, Collection
from django.conf import settings
from django.core.management.base import CommandError
from django.db import transaction
from typing_extensions import override
from zerver.lib.logging_util import log_to_file
from zerver.lib.management import ZulipBaseCommand
from zerver.models import UserProfile
from zproject.backends import ZulipLDAPError, sync_user_from_ldap
logger = logging.getLogger('zulip.sync_ldap_user_data')
log_to_file(logger, settings.LDAP_SYNC_LOG_PATH)

@transaction.atomic
def sync_ldap_user_data(user_profiles: Collection[UserProfile], deactivation_protection: bool=True) -> None:
    if False:
        i = 10
        return i + 15
    logger.info('Starting update.')
    try:
        realms = {u.realm.string_id for u in user_profiles}
        for u in user_profiles:
            try:
                sync_user_from_ldap(u, logger)
            except ZulipLDAPError as e:
                logger.error('Error attempting to update user %s:', u.delivery_email)
                logger.error(e.args[0])
        if deactivation_protection:
            if not UserProfile.objects.filter(is_bot=False, is_active=True).exists():
                raise Exception('LDAP sync would have deactivated all users. This is most likely due to a misconfiguration of LDAP settings. Rolling back...\nUse the --force option if the mass deactivation is intended.')
            for string_id in realms:
                if not UserProfile.objects.filter(is_bot=False, is_active=True, realm__string_id=string_id, role=UserProfile.ROLE_REALM_OWNER).exists():
                    raise Exception(f'LDAP sync would have deactivated all owners of realm {string_id}. This is most likely due to a misconfiguration of LDAP settings. Rolling back...\nUse the --force option if the mass deactivation is intended.')
    except Exception:
        logger.exception('LDAP sync failed')
        raise
    logger.info('Finished update.')

class Command(ZulipBaseCommand):

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        if False:
            while True:
                i = 10
        parser.add_argument('-f', '--force', action='store_true', help='Disable the protection against deactivating all users.')
        self.add_realm_args(parser)
        self.add_user_list_args(parser)

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if options.get('realm_id') is not None:
            realm = self.get_realm(options)
            user_profiles = self.get_users(options, realm, is_bot=False, include_deactivated=True)
        else:
            user_profiles = UserProfile.objects.select_related('realm').filter(is_bot=False)
            if not user_profiles.exists():
                raise CommandError('Zulip server contains no users. Have you created a realm?')
        if len(user_profiles) == 0:
            raise CommandError('Zulip server contains no users matching command-line parameters.')
        sync_ldap_user_data(user_profiles, not options['force'])