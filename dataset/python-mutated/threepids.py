import logging
import re
import typing
if typing.TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)
MAX_EMAIL_ADDRESS_LENGTH = 500

async def check_3pid_allowed(hs: 'HomeServer', medium: str, address: str, registration: bool=False) -> bool:
    """Checks whether a given format of 3PID is allowed to be used on this HS

    Args:
        hs: server
        medium: 3pid medium - e.g. email, msisdn
        address: address within that medium (e.g. "wotan@matrix.org")
            msisdns need to first have been canonicalised
        registration: whether we want to bind the 3PID as part of registering a new user.

    Returns:
        whether the 3PID medium/address is allowed to be added to this HS
    """
    if not await hs.get_password_auth_provider().is_3pid_allowed(medium, address, registration):
        return False
    if hs.config.registration.allowed_local_3pids:
        for constraint in hs.config.registration.allowed_local_3pids:
            logger.debug('Checking 3PID %s (%s) against %s (%s)', address, medium, constraint['pattern'], constraint['medium'])
            if medium == constraint['medium'] and re.match(constraint['pattern'], address):
                return True
    else:
        return True
    return False

def canonicalise_email(address: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    "'Canonicalise' email address\n    Case folding of local part of email address and lowercase domain part\n    See MSC2265, https://github.com/matrix-org/matrix-doc/pull/2265\n\n    Args:\n        address: email address to be canonicalised\n    Returns:\n        The canonical form of the email address\n    Raises:\n        ValueError if the address could not be parsed.\n    "
    address = address.strip()
    parts = address.split('@')
    if len(parts) != 2:
        logger.debug("Couldn't parse email address %s", address)
        raise ValueError('Unable to parse email address')
    return parts[0].casefold() + '@' + parts[1].lower()

def validate_email(address: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Does some basic validation on an email address.\n\n    Returns the canonicalised email, as returned by `canonicalise_email`.\n\n    Raises a ValueError if the email is invalid.\n    '
    address = canonicalise_email(address)
    if len(address) < 3:
        raise ValueError('Unable to parse email address')
    if len(address) > MAX_EMAIL_ADDRESS_LENGTH:
        raise ValueError('Unable to parse email address')
    return address