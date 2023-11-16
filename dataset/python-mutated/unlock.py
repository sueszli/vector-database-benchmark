import uuid
from os import environ
import requests
from decouple import config
from chalicelib.utils import helper
from chalicelib.utils.TimeUTC import TimeUTC

def __get_mid():
    if False:
        print('Hello World!')
    return str(uuid.UUID(int=uuid.getnode()))

def get_license():
    if False:
        for i in range(10):
            print('nop')
    return config('LICENSE_KEY', default='')

def check():
    if False:
        return 10
    license = get_license()
    if license is None or len(license) == 0:
        print('!! license key not found, please provide a LICENSE_KEY env var')
        environ['expiration'] = '-1'
        environ['numberOfSeats'] = '0'
        return
    print(f'validating: {helper.obfuscate(license)}')
    r = requests.post('https://api.openreplay.com/os/license', json={'mid': __get_mid(), 'license': get_license()})
    if r.status_code != 200 or 'errors' in r.json() or (not r.json()['data'].get('valid')):
        print('license validation failed')
        print(r.text)
        environ['expiration'] = '-1'
    else:
        environ['expiration'] = str(r.json()['data'].get('expiration'))
    environ['lastCheck'] = str(TimeUTC.now())
    if r.json()['data'].get('numberOfSeats') is not None:
        environ['numberOfSeats'] = str(r.json()['data']['numberOfSeats'])

def get_expiration_date():
    if False:
        for i in range(10):
            print('nop')
    return config('expiration', default=0, cast=int)

def is_valid():
    if False:
        return 10
    if config('lastCheck', default=None) is None or get_expiration_date() - TimeUTC.now() <= 0:
        check()
    return get_expiration_date() - TimeUTC.now() > 0