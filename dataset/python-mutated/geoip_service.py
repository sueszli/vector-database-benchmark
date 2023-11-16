"""
This is a tiny Flask app used for geoip lookups against a maxmind database.

If you are using this service be sure to set `geoip_location` in your ini file.

"""
import json
import GeoIP
from flask import Flask, make_response
application = Flask(__name__)
COUNTRY_DB_PATH = '/usr/share/GeoIP/GeoIP.dat'
CITY_DB_PATH = '/var/lib/GeoIP/GeoIPCity.dat'
ORG_DB_PATH = '/var/lib/GeoIP/GeoIPOrg.dat'
try:
    gc = GeoIP.open(COUNTRY_DB_PATH, GeoIP.GEOIP_MEMORY_CACHE)
except:
    gc = None
try:
    gi = GeoIP.open(CITY_DB_PATH, GeoIP.GEOIP_MEMORY_CACHE)
except:
    gi = None
try:
    go = GeoIP.open(ORG_DB_PATH, GeoIP.GEOIP_MEMORY_CACHE)
except:
    go = None

def json_response(result):
    if False:
        for i in range(10):
            print('nop')
    json_output = json.dumps(result, ensure_ascii=False, encoding='iso-8859-1')
    response = make_response(json_output.encode('utf-8'), 200)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

@application.route('/geoip/<ips>')
def get_record(ips):
    if False:
        return 10
    if gi:
        result = {ip: gi.record_by_addr(ip) for ip in ips.split('+')}
    elif gc:
        result = {ip: {'country_code': gc.country_code_by_addr(ip), 'country_name': gc.country_name_by_addr(ip)} for ip in ips.split('+')}
    else:
        result = {}
    return json_response(result)

@application.route('/org/<ips>')
def get_organizations(ips):
    if False:
        while True:
            i = 10
    if go:
        return json_response({ip: go.org_by_addr(ip) for ip in ips.split('+')})
    else:
        return json_response({})
if __name__ == '__main__':
    application.run()