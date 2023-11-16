import re
import geoip2.database
from app.config import Config
from .IPy import IP

def is_vaild_ip_target(ip):
    if False:
        for i in range(10):
            print('nop')
    if re.match('^\\d+\\.\\d+\\.\\d+\\.\\d+$|^\\d+\\.\\d+\\.\\d+\\.\\d+/\\d+$|^\\d+\\.\\d+\\.\\d+.\\d+-\\d+$', ip):
        return True
    else:
        return False

def transfer_ip_scope(target):
    if False:
        return 10
    '\n    将目标IP,IP段转换为合法的CIDR表示方法\n    '
    from . import get_logger
    logger = get_logger()
    try:
        return IP(target, make_net=True).strNormal(1)
    except Exception as e:
        logger.warn('error on ip_scope {} {}'.format(target, e))

def not_in_black_ips(target):
    if False:
        i = 10
        return i + 15
    from . import get_logger
    logger = get_logger()
    try:
        for ip in Config.BLACK_IPS:
            if '-' in target:
                target = target.split('-')[0]
            if '/' in target:
                target = target.split('/')[0]
            if IP(target) in IP(ip):
                return False
    except Exception as e:
        logger.warn('error on check black ip {} {}'.format(target, e))
    return True

def get_ip_asn(ip):
    if False:
        while True:
            i = 10
    from . import get_logger
    logger = get_logger()
    item = {}
    try:
        reader = geoip2.database.Reader(Config.GEOIP_ASN)
        response = reader.asn(ip)
        item['number'] = response.autonomous_system_number
        item['organization'] = response.autonomous_system_organization
        reader.close()
    except Exception as e:
        logger.warning('{} {}'.format(e, ip))
    return item

def get_ip_city(ip):
    if False:
        for i in range(10):
            print('nop')
    from . import get_logger
    logger = get_logger()
    try:
        reader = geoip2.database.Reader(Config.GEOIP_CITY)
        response = reader.city(ip)
        item = {'city': response.city.name, 'latitude': response.location.latitude, 'longitude': response.location.longitude, 'country_name': response.country.name, 'country_code': response.country.iso_code, 'region_name': response.subdivisions.most_specific.name, 'region_code': response.subdivisions.most_specific.iso_code}
        reader.close()
        return item
    except Exception as e:
        logger.warning('{} {}'.format(e, ip))
        return {}

def get_ip_type(ip):
    if False:
        print('Hello World!')
    from . import get_logger
    logger = get_logger()
    try:
        if ip.startswith('9.') or ip.startswith('11.'):
            return 'PRIVATE'
        ip_type = IP(ip).iptype()
        if ip_type in ['CARRIER_GRADE_NAT', 'LOOPBACK', 'RESERVED']:
            return 'PRIVATE'
        return ip_type
    except Exception as e:
        logger.warning('{} {}'.format(e, ip))
        return 'ERROR'

def ip_in_scope(ip, scope_list):
    if False:
        while True:
            i = 10
    from . import get_logger
    logger = get_logger()
    for item in scope_list:
        try:
            if IP(ip) in IP(item):
                return True
        except Exception as e:
            logger.warning('{} {} {}'.format(e, ip, item))