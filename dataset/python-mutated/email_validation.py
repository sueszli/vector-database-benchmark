from email.headerregistry import Address
from typing import Callable, Dict, Optional, Set, Tuple
from django.core import validators
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _
from zerver.lib.name_restrictions import is_disposable_domain
from zerver.models import DisposableEmailError, DomainNotAllowedForRealmError, EmailContainsPlusError, Realm, RealmDomain, get_users_by_delivery_email, is_cross_realm_bot_email

def validate_disposable(email: str) -> None:
    if False:
        print('Hello World!')
    if is_disposable_domain(Address(addr_spec=email).domain):
        raise DisposableEmailError

def get_realm_email_validator(realm: Realm) -> Callable[[str], None]:
    if False:
        for i in range(10):
            print('nop')
    if not realm.emails_restricted_to_domains:
        if realm.disallow_disposable_email_addresses:
            return validate_disposable
        return lambda email: None
    '\n    RESTRICTIVE REALMS:\n\n    Some realms only allow emails within a set\n    of domains that are configured in RealmDomain.\n\n    We get the set of domains up front so that\n    folks can validate multiple emails without\n    multiple round trips to the database.\n    '
    query = RealmDomain.objects.filter(realm=realm)
    rows = list(query.values('allow_subdomains', 'domain'))
    allowed_domains = {r['domain'] for r in rows}
    allowed_subdomains = {r['domain'] for r in rows if r['allow_subdomains']}

    def validate(email: str) -> None:
        if False:
            return 10
        '\n        We don\'t have to do a "disposable" check for restricted\n        domains, since the realm is already giving us\n        a small whitelist.\n        '
        address = Address(addr_spec=email)
        if '+' in address.username:
            raise EmailContainsPlusError
        domain = address.domain.lower()
        if domain in allowed_domains:
            return
        while len(domain) > 0:
            (subdomain, sep, domain) = domain.partition('.')
            if domain in allowed_subdomains:
                return
        raise DomainNotAllowedForRealmError
    return validate

def email_allowed_for_realm(email: str, realm: Realm) -> None:
    if False:
        print('Hello World!')
    '\n    Avoid calling this in a loop!\n    Instead, call get_realm_email_validator()\n    outside of the loop.\n    '
    get_realm_email_validator(realm)(email)

def validate_email_is_valid(email: str, validate_email_allowed_in_realm: Callable[[str], None]) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    try:
        validators.validate_email(email)
    except ValidationError:
        return _('Invalid address.')
    try:
        validate_email_allowed_in_realm(email)
    except DomainNotAllowedForRealmError:
        return _('Outside your domain.')
    except DisposableEmailError:
        return _('Please use your real email address.')
    except EmailContainsPlusError:
        return _('Email addresses containing + are not allowed.')
    return None

def email_reserved_for_system_bots_error(email: str) -> str:
    if False:
        while True:
            i = 10
    return f'{email} is reserved for system bots'

def get_existing_user_errors(target_realm: Realm, emails: Set[str], verbose: bool=False) -> Dict[str, Tuple[str, bool]]:
    if False:
        while True:
            i = 10
    '\n    We use this function even for a list of one emails.\n\n    It checks "new" emails to make sure that they don\'t\n    already exist.  There\'s a bit of fiddly logic related\n    to cross-realm bots and mirror dummies too.\n    '
    errors: Dict[str, Tuple[str, bool]] = {}
    users = get_users_by_delivery_email(emails, target_realm).only('delivery_email', 'is_active', 'is_mirror_dummy')
    "\n    A note on casing: We will preserve the casing used by\n    the user for email in most of this code.  The only\n    exception is when we do existence checks against\n    the `user_dict` dictionary.  (We don't allow two\n    users in the same realm to have the same effective\n    delivery email.)\n    "
    user_dict = {user.delivery_email.lower(): user for user in users}

    def process_email(email: str) -> None:
        if False:
            print('Hello World!')
        if is_cross_realm_bot_email(email):
            if verbose:
                msg = email_reserved_for_system_bots_error(email)
            else:
                msg = _('Reserved for system bots.')
            deactivated = False
            errors[email] = (msg, deactivated)
            return
        existing_user_profile = user_dict.get(email.lower())
        if existing_user_profile is None:
            return
        if existing_user_profile.is_mirror_dummy:
            if existing_user_profile.is_active:
                raise AssertionError('Mirror dummy user is already active!')
            return
        '\n        Email has already been taken by a "normal" user.\n        '
        deactivated = not existing_user_profile.is_active
        if existing_user_profile.is_active:
            if verbose:
                msg = _('{email} already has an account').format(email=email)
            else:
                msg = _('Already has an account.')
        else:
            msg = _('Account has been deactivated.')
        errors[email] = (msg, deactivated)
    for email in emails:
        process_email(email)
    return errors

def validate_email_not_already_in_realm(target_realm: Realm, email: str, verbose: bool=True) -> None:
    if False:
        return 10
    '\n    NOTE:\n        Only use this to validate that a single email\n        is not already used in the realm.\n\n        We should start using bulk_check_new_emails()\n        for any endpoint that takes multiple emails,\n        such as the "invite" interface.\n    '
    error_dict = get_existing_user_errors(target_realm, {email}, verbose)
    for (key, error_info) in error_dict.items():
        assert key == email
        (msg, deactivated) = error_info
        raise ValidationError(msg)