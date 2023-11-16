"""
Connection module for Amazon Route53

.. versionadded:: 2014.7.0

:configuration: This module accepts explicit route53 credentials but can also
    utilize IAM roles assigned to the instance through Instance Profiles.
    Dynamic credentials are then automatically obtained from AWS API and no
    further configuration is necessary. More Information available at:

    .. code-block:: yaml

        http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html

    If IAM roles are not used you need to specify them either in a pillar or
    in the minion's config file:

    .. code-block:: yaml

        route53.keyid: GKTADJGHEIQSXMKKRBJ08H
        route53.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

    A region may also be specified in the configuration:

    .. code-block:: yaml

        route53.region: us-east-1

    If a region is not specified, the default is 'universal', which is what the boto_route53
    library expects, rather than None.

    It's also possible to specify key, keyid and region via a profile, either
    as a passed in dict, or as a string to pull from pillars or minion config:

    .. code-block:: yaml

        myprofile:
          keyid: GKTADJGHEIQSXMKKRBJ08H
          key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
          region: us-east-1

:depends: boto
"""
import logging
import time
import salt.utils.compat
import salt.utils.odict as odict
import salt.utils.versions
from salt.exceptions import SaltInvocationError
log = logging.getLogger(__name__)
try:
    import boto
    import boto.route53
    import boto.route53.healthcheck
    from boto.route53.exception import DNSServerError
    logging.getLogger('boto').setLevel(logging.CRITICAL)
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load if boto libraries exist.\n    '
    return salt.utils.versions.check_boto_reqs(boto_ver='2.35.0', check_boto3=False)

def __init__(opts):
    if False:
        while True:
            i = 10
    if HAS_BOTO:
        __utils__['boto.assign_funcs'](__name__, 'route53', pack=__salt__)

def _get_split_zone(zone, _conn, private_zone):
    if False:
        print('Hello World!')
    '\n    With boto route53, zones can only be matched by name\n    or iterated over in a list.  Since the name will be the\n    same for public and private zones in a split DNS situation,\n    iterate over the list and match the zone name and public/private\n    status.\n    '
    for _zone in _conn.get_zones():
        if _zone.name == zone:
            _private_zone = True if _zone.config['PrivateZone'].lower() == 'true' else False
            if _private_zone == private_zone:
                return _zone
    return False

def _is_retryable_error(exception):
    if False:
        i = 10
        return i + 15
    return exception.code not in ['SignatureDoesNotMatch']

def describe_hosted_zones(zone_id=None, domain_name=None, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Return detailed info about one, or all, zones in the bound account.\n    If neither zone_id nor domain_name is provided, return all zones.\n    Note that the return format is slightly different between the \'all\'\n    and \'single\' description types.\n\n    zone_id\n        The unique identifier for the Hosted Zone\n\n    domain_name\n        The FQDN of the Hosted Zone (including final period)\n\n    region\n        Region to connect to.\n\n    key\n        Secret key to be used.\n\n    keyid\n        Access key to be used.\n\n    profile\n        A dict with region, key and keyid, or a pillar key (string) that\n        contains a dict with region, key and keyid.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.describe_hosted_zones domain_name=foo.bar.com.                 profile=\'{"region": "us-east-1", "keyid": "A12345678AB", "key": "xblahblahblah"}\'\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    if zone_id and domain_name:
        raise SaltInvocationError('At most one of zone_id or domain_name may be provided')
    retries = 10
    while retries:
        try:
            if zone_id:
                zone_id = zone_id.replace('/hostedzone/', '') if zone_id.startswith('/hostedzone/') else zone_id
                ret = getattr(conn.get_hosted_zone(zone_id), 'GetHostedZoneResponse', None)
            elif domain_name:
                ret = getattr(conn.get_hosted_zone_by_name(domain_name), 'GetHostedZoneResponse', None)
            else:
                marker = None
                ret = None
                while marker != '':
                    r = conn.get_all_hosted_zones(start_marker=marker, zone_list=ret)
                    ret = r['ListHostedZonesResponse']['HostedZones']
                    marker = r['ListHostedZonesResponse'].get('NextMarker', '')
            return ret if ret else []
        except DNSServerError as e:
            if retries:
                if 'Throttling' == e.code:
                    log.debug('Throttled by AWS API.')
                elif 'PriorRequestNotComplete' == e.code:
                    log.debug('The request was rejected by AWS API. Route 53 was still processing a prior request.')
                time.sleep(3)
                retries -= 1
                continue
            log.error('Could not list zones: %s', e.message)
            return []

def list_all_zones_by_name(region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    List, by their FQDNs, all hosted zones in the bound account.\n\n    region\n        Region to connect to.\n\n    key\n        Secret key to be used.\n\n    keyid\n        Access key to be used.\n\n    profile\n        A dict with region, key and keyid, or a pillar key (string) that\n        contains a dict with region, key and keyid.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.list_all_zones_by_name\n    '
    ret = describe_hosted_zones(region=region, key=key, keyid=keyid, profile=profile)
    return [r['Name'] for r in ret]

def list_all_zones_by_id(region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    List, by their IDs, all hosted zones in the bound account.\n\n    region\n        Region to connect to.\n\n    key\n        Secret key to be used.\n\n    keyid\n        Access key to be used.\n\n    profile\n        A dict with region, key and keyid, or a pillar key (string) that\n        contains a dict with region, key and keyid.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.list_all_zones_by_id\n    '
    ret = describe_hosted_zones(region=region, key=key, keyid=keyid, profile=profile)
    return [r['Id'].replace('/hostedzone/', '') for r in ret]

def zone_exists(zone, region=None, key=None, keyid=None, profile=None, retry_on_rate_limit=None, rate_limit_retries=None, retry_on_errors=True, error_retries=5):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check for the existence of a Route53 hosted zone.\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.zone_exists example.org\n\n    retry_on_errors\n        Continue to query if the zone exists after an error is\n        raised. The previously used argument `retry_on_rate_limit`\n        was deprecated for this argument. Users can still use\n        `retry_on_rate_limit` to ensure backwards compatibility,\n        but please migrate to using the favored `retry_on_errors`\n        argument instead.\n\n    error_retries\n        Number of times to attempt to query if the zone exists.\n        The previously used argument `rate_limit_retries` was\n        deprecated for this arguments. Users can still use\n        `rate_limit_retries` to ensure backwards compatibility,\n        but please migrate to using the favored `error_retries`\n        argument instead.\n    '
    if region is None:
        region = 'universal'
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    if retry_on_rate_limit or rate_limit_retries is not None:
        if retry_on_rate_limit is not None:
            retry_on_errors = retry_on_rate_limit
        if rate_limit_retries is not None:
            error_retries = rate_limit_retries
    while error_retries > 0:
        try:
            return bool(conn.get_zone(zone))
        except DNSServerError as e:
            if retry_on_errors and _is_retryable_error(e):
                if 'Throttling' == e.code:
                    log.debug('Throttled by AWS API.')
                elif 'PriorRequestNotComplete' == e.code:
                    log.debug('The request was rejected by AWS API. Route 53 was still processing a prior request ')
                time.sleep(3)
                error_retries -= 1
                continue
            raise
    return False

def create_zone(zone, private=False, vpc_id=None, vpc_region=None, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Create a Route53 hosted zone.\n\n    .. versionadded:: 2015.8.0\n\n    zone\n        DNS zone to create\n\n    private\n        True/False if the zone will be a private zone\n\n    vpc_id\n        VPC ID to associate the zone to (required if private is True)\n\n    vpc_region\n        VPC Region (required if private is True)\n\n    region\n        region endpoint to connect to\n\n    key\n        AWS key\n\n    keyid\n        AWS keyid\n\n    profile\n        AWS pillar profile\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.create_zone example.org\n    '
    if region is None:
        region = 'universal'
    if private:
        if not vpc_id or not vpc_region:
            msg = 'vpc_id and vpc_region must be specified for a private zone'
            raise SaltInvocationError(msg)
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    _zone = conn.get_zone(zone)
    if _zone:
        return False
    conn.create_zone(zone, private_zone=private, vpc_id=vpc_id, vpc_region=vpc_region)
    return True

def create_healthcheck(ip_addr=None, fqdn=None, region=None, key=None, keyid=None, profile=None, port=53, hc_type='TCP', resource_path='', string_match=None, request_interval=30, failure_threshold=3, retry_on_errors=True, error_retries=5):
    if False:
        print('Hello World!')
    '\n    Create a Route53 healthcheck\n\n    .. versionadded:: 2018.3.0\n\n    ip_addr\n\n        IP address to check.  ip_addr or fqdn is required.\n\n    fqdn\n\n        Domain name of the endpoint to check.  ip_addr or fqdn is required\n\n    port\n\n        Port to check\n\n    hc_type\n\n        Healthcheck type.  HTTP | HTTPS | HTTP_STR_MATCH | HTTPS_STR_MATCH | TCP\n\n    resource_path\n\n        Path to check\n\n    string_match\n\n        If hc_type is HTTP_STR_MATCH or HTTPS_STR_MATCH, the string to search for in the\n        response body from the specified resource\n\n    request_interval\n\n        The number of seconds between the time that Amazon Route 53 gets a response from\n        your endpoint and the time that it sends the next health-check request.\n\n    failure_threshold\n\n        The number of consecutive health checks that an endpoint must pass or fail for\n        Amazon Route 53 to change the current status of the endpoint from unhealthy to\n        healthy or vice versa.\n\n    region\n\n        Region endpoint to connect to\n\n    key\n\n        AWS key\n\n    keyid\n\n        AWS keyid\n\n    profile\n\n        AWS pillar profile\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.create_healthcheck 192.168.0.1\n        salt myminion boto_route53.create_healthcheck 192.168.0.1 port=443 hc_type=HTTPS                                                       resource_path=/ fqdn=blog.saltstack.furniture\n    '
    if fqdn is None and ip_addr is None:
        msg = 'One of the following must be specified: fqdn or ip_addr'
        log.error(msg)
        return {'error': msg}
    hc_ = boto.route53.healthcheck.HealthCheck(ip_addr, port, hc_type, resource_path, fqdn=fqdn, string_match=string_match, request_interval=request_interval, failure_threshold=failure_threshold)
    if region is None:
        region = 'universal'
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    while error_retries > 0:
        try:
            return {'result': conn.create_health_check(hc_)}
        except DNSServerError as exc:
            log.debug(exc)
            if retry_on_errors and _is_retryable_error(exc):
                if 'Throttling' == exc.code:
                    log.debug('Throttled by AWS API.')
                elif 'PriorRequestNotComplete' == exc.code:
                    log.debug('The request was rejected by AWS API. Route 53 was still processing a prior request.')
                time.sleep(3)
                error_retries -= 1
                continue
            return {'error': __utils__['boto.get_error'](exc)}
    return False

def delete_zone(zone, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Delete a Route53 hosted zone.\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.delete_zone example.org\n    '
    if region is None:
        region = 'universal'
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    _zone = conn.get_zone(zone)
    if _zone:
        conn.delete_hosted_zone(_zone.id)
        return True
    return False

def _encode_name(name):
    if False:
        print('Hello World!')
    return name.replace('*', '\\052')

def _decode_name(name):
    if False:
        while True:
            i = 10
    return name.replace('\\052', '*')

def get_record(name, zone, record_type, fetch_all=False, region=None, key=None, keyid=None, profile=None, split_dns=False, private_zone=False, identifier=None, retry_on_rate_limit=None, rate_limit_retries=None, retry_on_errors=True, error_retries=5):
    if False:
        return 10
    '\n    Get a record from a zone.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.get_record test.example.org example.org A\n\n    retry_on_errors\n        Continue to query if the zone exists after an error is\n        raised. The previously used argument `retry_on_rate_limit`\n        was deprecated for this argument. Users can still use\n        `retry_on_rate_limit` to ensure backwards compatibility,\n        but please migrate to using the favored `retry_on_errors`\n        argument instead.\n\n    error_retries\n        Number of times to attempt to query if the zone exists.\n        The previously used argument `rate_limit_retries` was\n        deprecated for this arguments. Users can still use\n        `rate_limit_retries` to ensure backwards compatibility,\n        but please migrate to using the favored `error_retries`\n        argument instead.\n    '
    if region is None:
        region = 'universal'
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    if retry_on_rate_limit or rate_limit_retries is not None:
        if retry_on_rate_limit is not None:
            retry_on_errors = retry_on_rate_limit
        if rate_limit_retries is not None:
            error_retries = rate_limit_retries
    _record = None
    ret = odict.OrderedDict()
    while error_retries > 0:
        try:
            if split_dns:
                _zone = _get_split_zone(zone, conn, private_zone)
            else:
                _zone = conn.get_zone(zone)
            if not _zone:
                msg = 'Failed to retrieve zone {}'.format(zone)
                log.error(msg)
                return None
            _type = record_type.upper()
            name = _encode_name(name)
            _record = _zone.find_records(name, _type, all=fetch_all, identifier=identifier)
            break
        except DNSServerError as e:
            if retry_on_errors and _is_retryable_error(e):
                if 'Throttling' == e.code:
                    log.debug('Throttled by AWS API.')
                elif 'PriorRequestNotComplete' == e.code:
                    log.debug('The request was rejected by AWS API. Route 53 was still processing a prior request.')
                time.sleep(3)
                error_retries -= 1
                continue
            raise
    if _record:
        ret['name'] = _decode_name(_record.name)
        ret['value'] = _record.resource_records[0]
        ret['record_type'] = _record.type
        ret['ttl'] = _record.ttl
        if _record.identifier:
            ret['identifier'] = []
            ret['identifier'].append(_record.identifier)
            ret['identifier'].append(_record.weight)
    return ret

def _munge_value(value, _type):
    if False:
        print('Hello World!')
    split_types = ['A', 'MX', 'AAAA', 'TXT', 'SRV', 'SPF', 'NS']
    if _type in split_types:
        return value.split(',')
    return value

def add_record(name, value, zone, record_type, identifier=None, ttl=None, region=None, key=None, keyid=None, profile=None, wait_for_sync=True, split_dns=False, private_zone=False, retry_on_rate_limit=None, rate_limit_retries=None, retry_on_errors=True, error_retries=5):
    if False:
        print('Hello World!')
    '\n    Add a record to a zone.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.add_record test.example.org 1.1.1.1 example.org A\n\n    retry_on_errors\n        Continue to query if the zone exists after an error is\n        raised. The previously used argument `retry_on_rate_limit`\n        was deprecated for this argument. Users can still use\n        `retry_on_rate_limit` to ensure backwards compatibility,\n        but please migrate to using the favored `retry_on_errors`\n        argument instead.\n\n    error_retries\n        Number of times to attempt to query if the zone exists.\n        The previously used argument `rate_limit_retries` was\n        deprecated for this arguments. Users can still use\n        `rate_limit_retries` to ensure backwards compatibility,\n        but please migrate to using the favored `error_retries`\n        argument instead.\n    '
    if region is None:
        region = 'universal'
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    if retry_on_rate_limit or rate_limit_retries is not None:
        if retry_on_rate_limit is not None:
            retry_on_errors = retry_on_rate_limit
        if rate_limit_retries is not None:
            error_retries = rate_limit_retries
    while error_retries > 0:
        try:
            if split_dns:
                _zone = _get_split_zone(zone, conn, private_zone)
            else:
                _zone = conn.get_zone(zone)
            if not _zone:
                msg = 'Failed to retrieve zone {}'.format(zone)
                log.error(msg)
                return False
            _type = record_type.upper()
            break
        except DNSServerError as e:
            if retry_on_errors and _is_retryable_error(e):
                if 'Throttling' == e.code:
                    log.debug('Throttled by AWS API.')
                elif 'PriorRequestNotComplete' == e.code:
                    log.debug('The request was rejected by AWS API. Route 53 was still processing a prior request.')
                time.sleep(3)
                error_retries -= 1
                continue
            raise
    _value = _munge_value(value, _type)
    while error_retries > 0:
        try:
            if ttl is None:
                ttl = 60
            status = _zone.add_record(_type, name, _value, ttl, identifier)
            return _wait_for_sync(status.id, conn, wait_for_sync)
        except DNSServerError as e:
            if retry_on_errors and _is_retryable_error(e):
                if 'Throttling' == e.code:
                    log.debug('Throttled by AWS API.')
                elif 'PriorRequestNotComplete' == e.code:
                    log.debug('The request was rejected by AWS API. Route 53 was still processing a prior request.')
                time.sleep(3)
                error_retries -= 1
                continue
            raise
    return False

def update_record(name, value, zone, record_type, identifier=None, ttl=None, region=None, key=None, keyid=None, profile=None, wait_for_sync=True, split_dns=False, private_zone=False, retry_on_rate_limit=None, rate_limit_retries=None, retry_on_errors=True, error_retries=5):
    if False:
        for i in range(10):
            print('nop')
    '\n    Modify a record in a zone.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.modify_record test.example.org 1.1.1.1 example.org A\n\n    retry_on_errors\n        Continue to query if the zone exists after an error is\n        raised. The previously used argument `retry_on_rate_limit`\n        was deprecated for this argument. Users can still use\n        `retry_on_rate_limit` to ensure backwards compatibility,\n        but please migrate to using the favored `retry_on_errors`\n        argument instead.\n\n    error_retries\n        Number of times to attempt to query if the zone exists.\n        The previously used argument `rate_limit_retries` was\n        deprecated for this arguments. Users can still use\n        `rate_limit_retries` to ensure backwards compatibility,\n        but please migrate to using the favored `error_retries`\n        argument instead.\n    '
    if region is None:
        region = 'universal'
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    if split_dns:
        _zone = _get_split_zone(zone, conn, private_zone)
    else:
        _zone = conn.get_zone(zone)
    if not _zone:
        msg = 'Failed to retrieve zone {}'.format(zone)
        log.error(msg)
        return False
    _type = record_type.upper()
    if retry_on_rate_limit or rate_limit_retries is not None:
        if retry_on_rate_limit is not None:
            retry_on_errors = retry_on_rate_limit
        if rate_limit_retries is not None:
            error_retries = rate_limit_retries
    _value = _munge_value(value, _type)
    while error_retries > 0:
        try:
            old_record = _zone.find_records(name, _type, identifier=identifier)
            if not old_record:
                return False
            status = _zone.update_record(old_record, _value, ttl, identifier)
            return _wait_for_sync(status.id, conn, wait_for_sync)
        except DNSServerError as e:
            if retry_on_errors and _is_retryable_error(e):
                if 'Throttling' == e.code:
                    log.debug('Throttled by AWS API.')
                elif 'PriorRequestNotComplete' == e.code:
                    log.debug('The request was rejected by AWS API. Route 53 was still processing a prior request.')
                time.sleep(3)
                error_retries -= 1
                continue
            raise
    return False

def delete_record(name, zone, record_type, identifier=None, all_records=False, region=None, key=None, keyid=None, profile=None, wait_for_sync=True, split_dns=False, private_zone=False, retry_on_rate_limit=None, rate_limit_retries=None, retry_on_errors=True, error_retries=5):
    if False:
        return 10
    '\n    Modify a record in a zone.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.delete_record test.example.org example.org A\n\n    retry_on_errors\n        Continue to query if the zone exists after an error is\n        raised. The previously used argument `retry_on_rate_limit`\n        was deprecated for this argument. Users can still use\n        `retry_on_rate_limit` to ensure backwards compatibility,\n        but please migrate to using the favored `retry_on_errors`\n        argument instead.\n\n    error_retries\n        Number of times to attempt to query if the zone exists.\n        The previously used argument `rate_limit_retries` was\n        deprecated for this arguments. Users can still use\n        `rate_limit_retries` to ensure backwards compatibility,\n        but please migrate to using the favored `error_retries`\n        argument instead.\n    '
    if region is None:
        region = 'universal'
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    if split_dns:
        _zone = _get_split_zone(zone, conn, private_zone)
    else:
        _zone = conn.get_zone(zone)
    if not _zone:
        msg = 'Failed to retrieve zone {}'.format(zone)
        log.error(msg)
        return False
    _type = record_type.upper()
    if retry_on_rate_limit or rate_limit_retries is not None:
        if retry_on_rate_limit is not None:
            retry_on_errors = retry_on_rate_limit
        if rate_limit_retries is not None:
            error_retries = rate_limit_retries
    while error_retries > 0:
        try:
            old_record = _zone.find_records(name, _type, all=all_records, identifier=identifier)
            if not old_record:
                return False
            status = _zone.delete_record(old_record)
            return _wait_for_sync(status.id, conn, wait_for_sync)
        except DNSServerError as e:
            if retry_on_errors and _is_retryable_error(e):
                if 'Throttling' == e.code:
                    log.debug('Throttled by AWS API.')
                elif 'PriorRequestNotComplete' == e.code:
                    log.debug('The request was rejected by AWS API. Route 53 was still processing a prior request.')
                time.sleep(3)
                error_retries -= 1
                continue
            raise

def _try_func(conn, func, **args):
    if False:
        return 10
    tries = 30
    while True:
        try:
            return getattr(conn, func)(**args)
        except AttributeError as e:
            log.error('Function `%s()` not found for AWS connection object %s', func, conn)
            return None
        except DNSServerError as e:
            if tries and e.code == 'Throttling':
                log.debug('Throttled by AWS API.  Will retry in 5 seconds')
                time.sleep(5)
                tries -= 1
                continue
            log.error('Failed calling %s(): %s', func, e)
            return None

def _wait_for_sync(status, conn, wait=True):
    if False:
        print('Hello World!')
    if wait is True:
        wait = 600
    if not wait:
        return True
    orig_wait = wait
    log.info('Waiting up to %s seconds for Route53 changes to synchronize', orig_wait)
    while wait > 0:
        change = conn.get_change(status)
        current = change.GetChangeResponse.ChangeInfo.Status
        if current == 'INSYNC':
            return True
        sleep = wait if wait % 60 == wait else 60
        log.info('Sleeping %s seconds waiting for changes to synch (current status %s)', sleep, current)
        time.sleep(sleep)
        wait -= sleep
        continue
    log.error('Route53 changes not synced after %s seconds.', orig_wait)
    return False

def create_hosted_zone(domain_name, caller_ref=None, comment='', private_zone=False, vpc_id=None, vpc_name=None, vpc_region=None, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a new Route53 Hosted Zone. Returns a Python data structure with information about the\n    newly created Hosted Zone.\n\n    domain_name\n        The name of the domain. This must be fully-qualified, terminating with a period.  This is\n        the name you have registered with your domain registrar.  It is also the name you will\n        delegate from your registrar to the Amazon Route 53 delegation servers returned in response\n        to this request.\n\n    caller_ref\n        A unique string that identifies the request and that allows create_hosted_zone() calls to\n        be retried without the risk of executing the operation twice.  It can take several minutes\n        for the change to replicate globally, and change from PENDING to INSYNC status. Thus it's\n        best to provide some value for this where possible, since duplicate calls while the first\n        is in PENDING status will be accepted and can lead to multiple copies of the zone being\n        created.  On the other hand, if a zone is created with a given caller_ref, then deleted,\n        a second attempt to create a zone with the same caller_ref will fail until that caller_ref\n        is flushed from the Route53 system, which can take upwards of 24 hours.\n\n    comment\n        Any comments you want to include about the hosted zone.\n\n    private_zone\n        Set True if creating a private hosted zone.\n\n    vpc_id\n        When creating a private hosted zone, either the VPC ID or VPC Name to associate with is\n        required.  Exclusive with vpe_name.  Ignored when creating a non-private zone.\n\n    vpc_name\n        When creating a private hosted zone, either the VPC ID or VPC Name to associate with is\n        required.  Exclusive with vpe_id.  Ignored when creating a non-private zone.\n\n    vpc_region\n        When creating a private hosted zone, the region of the associated VPC is required.  If not\n        provided, an effort will be made to determine it from vpc_id or vpc_name, where possible.\n        If this fails, you'll need to provide an explicit value for this option.  Ignored when\n        creating a non-private zone.\n\n    region\n        Region endpoint to connect to.\n\n    key\n        AWS key to bind with.\n\n    keyid\n        AWS keyid to bind with.\n\n    profile\n        Dict, or pillar key pointing to a dict, containing AWS region/key/keyid.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_route53.create_hosted_zone example.org\n    "
    if region is None:
        region = 'universal'
    if not domain_name.endswith('.'):
        raise SaltInvocationError('Domain MUST be fully-qualified, complete with ending period.')
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    deets = conn.get_hosted_zone_by_name(domain_name)
    if deets:
        log.info('Route53 hosted zone %s already exists', domain_name)
        return None
    args = {'domain_name': domain_name, 'caller_ref': caller_ref, 'comment': comment, 'private_zone': private_zone}
    if private_zone:
        if not _exactly_one((vpc_name, vpc_id)):
            raise SaltInvocationError('Either vpc_name or vpc_id is required when creating a private zone.')
        vpcs = __salt__['boto_vpc.describe_vpcs'](vpc_id=vpc_id, name=vpc_name, region=region, key=key, keyid=keyid, profile=profile).get('vpcs', [])
        if vpc_region and vpcs:
            vpcs = [v for v in vpcs if v['region'] == vpc_region]
        if not vpcs:
            log.error('Private zone requested but a VPC matching given criteria not found.')
            return None
        if len(vpcs) > 1:
            log.error('Private zone requested but multiple VPCs matching given criteria found: %s.', [v['id'] for v in vpcs])
            return None
        vpc = vpcs[0]
        if vpc_name:
            vpc_id = vpc['id']
        if not vpc_region:
            vpc_region = vpc['region']
        args.update({'vpc_id': vpc_id, 'vpc_region': vpc_region})
    elif any((vpc_id, vpc_name, vpc_region)):
        log.info('Options vpc_id, vpc_name, and vpc_region are ignored when creating non-private zones.')
    r = _try_func(conn, 'create_hosted_zone', **args)
    if r is None:
        log.error('Failed to create hosted zone %s', domain_name)
        return None
    r = r.get('CreateHostedZoneResponse', {})
    status = r.pop('ChangeInfo', {}).get('Id', '').replace('/change/', '')
    synced = _wait_for_sync(status, conn, wait=600)
    if not synced:
        log.error('Hosted zone %s not synced after 600 seconds.', domain_name)
        return None
    return r