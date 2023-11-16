from argparse import ArgumentParser
from typing import Any
from django.conf import settings
from django.core.management.base import CommandError
from typing_extensions import override
from zerver.actions.realm_settings import do_delete_all_realm_attachments
from zerver.lib.management import ZulipBaseCommand
from zerver.models import Message, UserProfile

class Command(ZulipBaseCommand):
    help = 'Script to permanently delete a realm. Recommended only for removing\nrealms used for testing; consider using deactivate_realm instead.'

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        if False:
            while True:
                i = 10
        self.add_realm_args(parser, required=True)

    @override
    def handle(self, *args: Any, **options: str) -> None:
        if False:
            print('Hello World!')
        realm = self.get_realm(options)
        assert realm is not None
        user_count = UserProfile.objects.filter(realm_id=realm.id, is_active=True, is_bot=False).count()
        message_count = Message.objects.filter(realm=realm).count()
        print(f'This realm has {user_count} users and {message_count} messages.\n')
        if settings.BILLING_ENABLED:
            from corporate.models import CustomerPlan, get_customer_by_realm
            customer = get_customer_by_realm(realm)
            if customer and (customer.stripe_customer_id or CustomerPlan.objects.filter(customer=customer).exists()):
                raise CommandError('This realm has had a billing relationship associated with it!')
        print('This command will \x1b[91mPERMANENTLY DELETE\x1b[0m all data for this realm.  Most use cases will be better served by scrub_realm and/or deactivate_realm.')
        confirmation = input('Type the name of the realm to confirm: ')
        if confirmation != realm.string_id:
            raise CommandError('Aborting!')
        do_delete_all_realm_attachments(realm)
        realm.delete()
        print('Realm has been successfully permanently deleted.')