"""Performs requests to the Google Maps Address Validation API."""
from googlemaps import exceptions
_ADDRESSVALIDATION_BASE_URL = 'https://addressvalidation.googleapis.com'

def _addressvalidation_extract(response):
    if False:
        for i in range(10):
            print('nop')
    '\n    Mimics the exception handling logic in ``client._get_body``, but\n    for addressvalidation which uses a different response format.\n    '
    body = response.json()
    return body

def addressvalidation(client, addressLines, regionCode=None, locality=None, enableUspsCass=None):
    if False:
        i = 10
        return i + 15
    '\n    The Google Maps Address Validation API returns a verification of an address\n    See https://developers.google.com/maps/documentation/address-validation/overview\n    request must include parameters below.\n    :param addressLines: The address to validate\n    :type addressLines: array \n    :param regionCode: (optional) The country code\n    :type regionCode: string  \n    :param locality: (optional) Restrict to a locality, ie:Mountain View\n    :type locality: string\n    :param enableUspsCass For the "US" and "PR" regions only, you can optionally enable the Coding Accuracy Support System (CASS) from the United States Postal Service (USPS)\n    :type locality: boolean\n    '
    params = {'address': {'addressLines': addressLines}}
    if regionCode is not None:
        params['address']['regionCode'] = regionCode
    if locality is not None:
        params['address']['locality'] = locality
    if enableUspsCass is not False or enableUspsCass is not None:
        params['enableUspsCass'] = enableUspsCass
    return client._request('/v1:validateAddress', {}, base_url=_ADDRESSVALIDATION_BASE_URL, extract_body=_addressvalidation_extract, post_json=params)