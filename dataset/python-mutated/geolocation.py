"""Performs requests to the Google Maps Geolocation API."""
from googlemaps import exceptions
_GEOLOCATION_BASE_URL = 'https://www.googleapis.com'

def _geolocation_extract(response):
    if False:
        return 10
    '\n    Mimics the exception handling logic in ``client._get_body``, but\n    for geolocation which uses a different response format.\n    '
    body = response.json()
    if response.status_code in (200, 404):
        return body
    try:
        error = body['error']['errors'][0]['reason']
    except KeyError:
        error = None
    if response.status_code == 403:
        raise exceptions._OverQueryLimit(response.status_code, error)
    else:
        raise exceptions.ApiError(response.status_code, error)

def geolocate(client, home_mobile_country_code=None, home_mobile_network_code=None, radio_type=None, carrier=None, consider_ip=None, cell_towers=None, wifi_access_points=None):
    if False:
        print('Hello World!')
    "\n    The Google Maps Geolocation API returns a location and accuracy\n    radius based on information about cell towers and WiFi nodes given.\n\n    See https://developers.google.com/maps/documentation/geolocation/intro\n    for more info, including more detail for each parameter below.\n\n    :param home_mobile_country_code: The mobile country code (MCC) for\n        the device's home network.\n    :type home_mobile_country_code: string\n\n    :param home_mobile_network_code: The mobile network code (MCC) for\n        the device's home network.\n    :type home_mobile_network_code: string\n\n    :param radio_type: The mobile radio type. Supported values are\n        lte, gsm, cdma, and wcdma. While this field is optional, it\n        should be included if a value is available, for more accurate\n        results.\n    :type radio_type: string\n\n    :param carrier: The carrier name.\n    :type carrier: string\n\n    :param consider_ip: Specifies whether to fall back to IP geolocation\n        if wifi and cell tower signals are not available. Note that the\n        IP address in the request header may not be the IP of the device.\n    :type consider_ip: bool\n\n    :param cell_towers: A list of cell tower dicts. See\n        https://developers.google.com/maps/documentation/geolocation/intro#cell_tower_object\n        for more detail.\n    :type cell_towers: list of dicts\n\n    :param wifi_access_points: A list of WiFi access point dicts. See\n        https://developers.google.com/maps/documentation/geolocation/intro#wifi_access_point_object\n        for more detail.\n    :type wifi_access_points: list of dicts\n    "
    params = {}
    if home_mobile_country_code is not None:
        params['homeMobileCountryCode'] = home_mobile_country_code
    if home_mobile_network_code is not None:
        params['homeMobileNetworkCode'] = home_mobile_network_code
    if radio_type is not None:
        params['radioType'] = radio_type
    if carrier is not None:
        params['carrier'] = carrier
    if consider_ip is not None:
        params['considerIp'] = consider_ip
    if cell_towers is not None:
        params['cellTowers'] = cell_towers
    if wifi_access_points is not None:
        params['wifiAccessPoints'] = wifi_access_points
    return client._request('/geolocation/v1/geolocate', {}, base_url=_GEOLOCATION_BASE_URL, extract_body=_geolocation_extract, post_json=params)