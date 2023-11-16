"""
OnionShare | https://onionshare.org/

Copyright (C) 2014-2022 Micah Lee, et al. <micah@micahflee.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import requests

class CensorshipCircumventionError(Exception):
    """
    There was a problem connecting to the Tor CensorshipCircumvention API.
    """

class CensorshipCircumvention(object):
    """
    Connect to the Tor Moat APIs to retrieve censorship
    circumvention recommendations or the latest bridges.

    We support reaching this API over Tor, or Meek
    (domain fronting) if Tor is not connected.
    """

    def __init__(self, common, meek=None, onion=None):
        if False:
            print('Hello World!')
        '\n        Set up the CensorshipCircumvention object to hold\n        common and meek objects.\n        '
        self.common = common
        self.common.log('CensorshipCircumvention', '__init__')
        self.api_proxies = {}
        if meek:
            self.meek = meek
            self.common.log('CensorshipCircumvention', '__init__', 'Using Meek with CensorshipCircumvention API')
            self.api_proxies = self.meek.meek_proxies
        if onion:
            self.onion = onion
            if not self.onion.is_authenticated:
                return False
            else:
                self.common.log('CensorshipCircumvention', '__init__', 'Using Tor with CensorshipCircumvention API')
                (socks_address, socks_port) = self.onion.get_tor_socks_port()
                self.api_proxies = {'http': f'socks5h://{socks_address}:{socks_port}', 'https': f'socks5h://{socks_address}:{socks_port}'}

    def request_map(self, country=False):
        if False:
            print('Hello World!')
        "\n        Retrieves the Circumvention map from Tor Project and store it\n        locally for further look-ups if required.\n\n        Optionally pass a country code in order to get recommended settings\n        just for that country.\n\n        Note that this API endpoint doesn't return actual bridges,\n        it just returns the recommended bridge type countries.\n        "
        self.common.log('CensorshipCircumvention', 'request_map', f'country={country}')
        if not self.api_proxies:
            return False
        endpoint = 'https://bridges.torproject.org/moat/circumvention/map'
        data = {}
        if country:
            data = {'country': country}
        try:
            r = requests.post(endpoint, json=data, headers={'Content-Type': 'application/vnd.api+json'}, proxies=self.api_proxies)
            if r.status_code != 200:
                self.common.log('CensorshipCircumvention', 'request_map', f'status_code={r.status_code}')
                return False
            result = r.json()
            if 'errors' in result:
                self.common.log('CensorshipCircumvention', 'request_map', f"errors={result['errors']}")
                return False
            return result
        except requests.exceptions.RequestException as e:
            raise CensorshipCircumventionError(e)

    def request_settings(self, country=False, transports=False):
        if False:
            print('Hello World!')
        '\n        Retrieves the Circumvention Settings from Tor Project, which\n        will return recommended settings based on the country code of\n        the requesting IP.\n\n        Optionally, a country code can be specified in order to override\n        the IP detection.\n\n        Optionally, a list of transports can be specified in order to\n        return recommended settings for just that transport type.\n        '
        self.common.log('CensorshipCircumvention', 'request_settings', f'country={country}, transports={transports}')
        if not self.api_proxies:
            return False
        endpoint = 'https://bridges.torproject.org/moat/circumvention/settings'
        data = {}
        if country:
            self.common.log('CensorshipCircumvention', 'request_settings', f'Trying to obtain bridges for country={country}')
            data = {'country': country}
        if transports:
            data.append({'transports': transports})
        try:
            r = requests.post(endpoint, json=data, headers={'Content-Type': 'application/vnd.api+json'}, proxies=self.api_proxies)
            if r.status_code != 200:
                self.common.log('CensorshipCircumvention', 'request_settings', f'status_code={r.status_code}')
                return False
            result = r.json()
            self.common.log('CensorshipCircumvention', 'request_settings', f'result={result}')
            if 'errors' in result:
                self.common.log('CensorshipCircumvention', 'request_settings', f"errors={result['errors']}")
                return False
            if not 'settings' in result or result['settings'] is None:
                self.common.log('CensorshipCircumvention', 'request_settings', 'No settings found for this country')
                return False
            return result
        except requests.exceptions.RequestException as e:
            raise CensorshipCircumventionError(e)

    def request_builtin_bridges(self):
        if False:
            i = 10
            return i + 15
        '\n        Retrieves the list of built-in bridges from the Tor Project.\n        '
        if not self.api_proxies:
            return False
        endpoint = 'https://bridges.torproject.org/moat/circumvention/builtin'
        try:
            r = requests.post(endpoint, headers={'Content-Type': 'application/vnd.api+json'}, proxies=self.api_proxies)
            if r.status_code != 200:
                self.common.log('CensorshipCircumvention', 'request_builtin_bridges', f'status_code={r.status_code}')
                return False
            result = r.json()
            if 'errors' in result:
                self.common.log('CensorshipCircumvention', 'request_builtin_bridges', f"errors={result['errors']}")
                return False
            return result
        except requests.exceptions.RequestException as e:
            raise CensorshipCircumventionError(e)

    def save_settings(self, settings, bridge_settings):
        if False:
            return 10
        '\n        Checks the bridges and saves them in settings.\n        '
        self.common.log('CensorshipCircumvention', 'save_settings', f'bridge_settings: {bridge_settings}')
        bridges_ok = False
        self.settings = settings
        bridges = bridge_settings['settings'][0]['bridges']
        bridge_strings = bridges['bridge_strings']
        self.settings.set('bridges_type', 'custom')
        bridges_checked = self.common.check_bridges_valid(bridge_strings)
        if bridges_checked:
            self.settings.set('bridges_custom', '\n'.join(bridges_checked))
            bridges_ok = True
        if bridges_ok:
            self.common.log('CensorshipCircumvention', 'save_settings', 'Saving settings with automatically-obtained bridges')
            self.settings.set('bridges_enabled', True)
            self.settings.save()
            return True
        else:
            self.common.log('CensorshipCircumvention', 'save_settings', 'Could not use any of the obtained bridges.')
            return False

    def request_default_bridges(self):
        if False:
            return 10
        '\n        Retrieves the list of default fall-back bridges from the Tor Project.\n\n        These are intended for when no censorship settings were found for a\n        specific country, but maybe there was some connection issue anyway.\n        '
        if not self.api_proxies:
            return False
        endpoint = 'https://bridges.torproject.org/moat/circumvention/defaults'
        try:
            r = requests.get(endpoint, headers={'Content-Type': 'application/vnd.api+json'}, proxies=self.api_proxies)
            if r.status_code != 200:
                self.common.log('CensorshipCircumvention', 'request_default_bridges', f'status_code={r.status_code}')
                return False
            result = r.json()
            if 'errors' in result:
                self.common.log('CensorshipCircumvention', 'request_default_bridges', f"errors={result['errors']}")
                return False
            return result
        except requests.exceptions.RequestException as e:
            raise CensorshipCircumventionError(e)