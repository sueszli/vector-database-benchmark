from argparse import ArgumentParser
from typing import Any

from django.core.management.base import CommandError
from typing_extensions import override

from zerver.actions.users import (
    do_change_can_create_users,
    do_change_can_forge_sender,
    do_change_is_billing_admin,
    do_change_user_role,
)
from zerver.lib.management import ZulipBaseCommand
from zerver.models import UserProfile


class Command(ZulipBaseCommand):
    help = """Change role of an existing user in their (own) Realm.

ONLY perform this on customer request from an authorized person.
"""

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("email", metavar="<email>", help="email of user to change role")
        parser.add_argument(
            "new_role",
            metavar="<new_role>",
            choices=[
                "owner",
                "admin",
                "moderator",
                "member",
                "guest",
                "can_forge_sender",
                "can_create_users",
                "is_billing_admin",
            ],
            help="new role of the user",
        )
        parser.add_argument(
            "--revoke",
            dest="grant",
            action="store_false",
            help="Remove can_forge_sender or can_create_users permission.",
        )
        self.add_realm_args(parser, required=True)

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        email = options["email"]
        realm = self.get_realm(options)

        user = self.get_user(email, realm)

        user_role_map = {
            "owner": UserProfile.ROLE_REALM_OWNER,
            "admin": UserProfile.ROLE_REALM_ADMINISTRATOR,
            "moderator": UserProfile.ROLE_MODERATOR,
            "member": UserProfile.ROLE_MEMBER,
            "guest": UserProfile.ROLE_GUEST,
        }

        if options["new_role"] not in ["can_forge_sender", "can_create_users", "is_billing_admin"]:
            new_role = user_role_map[options["new_role"]]
            if not options["grant"]:
                raise CommandError(
                    "Revoke not supported with this permission; please specify new role."
                )
            if new_role == user.role:
                raise CommandError("User already has this role.")
            old_role_name = UserProfile.ROLE_ID_TO_NAME_MAP[user.role]
            do_change_user_role(user, new_role, acting_user=None)
            new_role_name = UserProfile.ROLE_ID_TO_NAME_MAP[user.role]
            print(
                f"Role for {user.delivery_email} changed from {old_role_name} to {new_role_name}."
            )
            return

        if options["new_role"] == "can_forge_sender":
            if user.can_forge_sender and options["grant"]:
                raise CommandError("User can already forge messages for this realm.")
            elif not user.can_forge_sender and not options["grant"]:
                raise CommandError("User can't forge messages for this realm.")
            do_change_can_forge_sender(user, options["grant"])

            granted_text = "have" if options["grant"] else "not have"
            print(
                f"{user.delivery_email} changed to {granted_text} {options['new_role']} permission."
            )
        elif options["new_role"] == "can_create_users":
            if user.can_create_users and options["grant"]:
                raise CommandError("User can already create users for this realm.")
            elif not user.can_create_users and not options["grant"]:
                raise CommandError("User can't create users for this realm.")
            do_change_can_create_users(user, options["grant"])
        else:
            assert options["new_role"] == "is_billing_admin"
            if user.is_billing_admin and options["grant"]:
                raise CommandError("User already is a billing admin for this realm.")
            elif not user.is_billing_admin and not options["grant"]:
                raise CommandError("User is not a billing admin for this realm.")

            do_change_is_billing_admin(user, options["grant"])
