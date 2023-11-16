import phonenumbers
from synapse.api.errors import SynapseError

def phone_number_to_msisdn(country: str, number: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Takes an ISO-3166-1 2 letter country code and phone number and\n    returns an msisdn representing the canonical version of that\n    phone number.\n\n    As an example, if `country` is "GB" and `number` is "7470674927", this\n    function will return "447470674927".\n\n    Args:\n        country: ISO-3166-1 2 letter country code\n        number: Phone number in a national or international format\n\n    Returns:\n        The canonical form of the phone number, as an msisdn.\n    Raises:\n        SynapseError if the number could not be parsed.\n    '
    try:
        phoneNumber = phonenumbers.parse(number, country)
    except phonenumbers.NumberParseException:
        raise SynapseError(400, 'Unable to parse phone number')
    return phonenumbers.format_number(phoneNumber, phonenumbers.PhoneNumberFormat.E164)[1:]