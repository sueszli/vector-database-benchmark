from __future__ import absolute_import
import sys
import time
import random
from collections import defaultdict
from datetime import datetime
from logging import debug, info, warning, error
try:
    import xml.etree.ElementTree as ET
except ImportError:
    import elementtree.ElementTree as ET
from .S3 import S3
from .Config import Config
from .Exceptions import CloudFrontError, ParameterError
from .ExitCodes import EX_OK, EX_GENERAL, EX_PARTIAL
from .BaseUtils import getTreeFromXml, appendXmlTextNode, getDictFromTree, dateS3toPython, encode_to_s3, decode_from_s3
from .Utils import getBucketFromHostname, getHostnameFromBucket, deunicodise, convertHeaderTupleListToDict
from .Crypto import sign_string_v2
from .S3Uri import S3Uri, S3UriS3
from .ConnMan import ConnMan
from .SortedDict import SortedDict
PY3 = sys.version_info >= (3, 0)
cloudfront_api_version = '2010-11-01'
cloudfront_resource = '/%(api_ver)s/distribution' % {'api_ver': cloudfront_api_version}

def output(message):
    if False:
        i = 10
        return i + 15
    sys.stdout.write(message + '\n')

def pretty_output(label, message):
    if False:
        while True:
            i = 10
    label = ('%s:' % label).ljust(15)
    output('%s %s' % (label, message))

class DistributionSummary(object):

    def __init__(self, tree):
        if False:
            return 10
        if tree.tag != 'DistributionSummary':
            raise ValueError('Expected <DistributionSummary /> xml, got: <%s />' % tree.tag)
        self.parse(tree)

    def parse(self, tree):
        if False:
            return 10
        self.info = getDictFromTree(tree)
        self.info['Enabled'] = self.info['Enabled'].lower() == 'true'
        if 'CNAME' in self.info and type(self.info['CNAME']) != list:
            self.info['CNAME'] = [self.info['CNAME']]

    def uri(self):
        if False:
            return 10
        return S3Uri(u'cf://%s' % self.info['Id'])

class DistributionList(object):

    def __init__(self, xml):
        if False:
            while True:
                i = 10
        tree = getTreeFromXml(xml)
        if tree.tag != 'DistributionList':
            raise ValueError('Expected <DistributionList /> xml, got: <%s />' % tree.tag)
        self.parse(tree)

    def parse(self, tree):
        if False:
            i = 10
            return i + 15
        self.info = getDictFromTree(tree)
        self.info['IsTruncated'] = self.info['IsTruncated'].lower() == 'true'
        self.dist_summs = []
        for dist_summ in tree.findall('.//DistributionSummary'):
            self.dist_summs.append(DistributionSummary(dist_summ))

class Distribution(object):

    def __init__(self, xml):
        if False:
            while True:
                i = 10
        tree = getTreeFromXml(xml)
        if tree.tag != 'Distribution':
            raise ValueError('Expected <Distribution /> xml, got: <%s />' % tree.tag)
        self.parse(tree)

    def parse(self, tree):
        if False:
            while True:
                i = 10
        self.info = getDictFromTree(tree)
        self.info['LastModifiedTime'] = dateS3toPython(self.info['LastModifiedTime'])
        self.info['DistributionConfig'] = DistributionConfig(tree=tree.find('.//DistributionConfig'))

    def uri(self):
        if False:
            return 10
        return S3Uri(u'cf://%s' % self.info['Id'])

class DistributionConfig(object):
    EMPTY_CONFIG = '<DistributionConfig><S3Origin><DNSName/></S3Origin><CallerReference/><Enabled>true</Enabled></DistributionConfig>'
    xmlns = 'http://cloudfront.amazonaws.com/doc/%(api_ver)s/' % {'api_ver': cloudfront_api_version}

    def __init__(self, xml=None, tree=None):
        if False:
            print('Hello World!')
        if xml is None:
            xml = DistributionConfig.EMPTY_CONFIG
        if tree is None:
            tree = getTreeFromXml(xml)
        if tree.tag != 'DistributionConfig':
            raise ValueError('Expected <DistributionConfig /> xml, got: <%s />' % tree.tag)
        self.parse(tree)

    def parse(self, tree):
        if False:
            while True:
                i = 10
        self.info = getDictFromTree(tree)
        self.info['Enabled'] = self.info['Enabled'].lower() == 'true'
        if 'CNAME' not in self.info:
            self.info['CNAME'] = []
        if type(self.info['CNAME']) != list:
            self.info['CNAME'] = [self.info['CNAME']]
        self.info['CNAME'] = [cname.lower() for cname in self.info['CNAME']]
        if 'Comment' not in self.info:
            self.info['Comment'] = ''
        if 'DefaultRootObject' not in self.info:
            self.info['DefaultRootObject'] = ''
        logging_nodes = tree.findall('.//Logging')
        if logging_nodes:
            logging_dict = getDictFromTree(logging_nodes[0])
            (logging_dict['Bucket'], success) = getBucketFromHostname(logging_dict['Bucket'])
            if not success:
                warning('Logging to unparsable bucket name: %s' % logging_dict['Bucket'])
            self.info['Logging'] = S3UriS3(u's3://%(Bucket)s/%(Prefix)s' % logging_dict)
        else:
            self.info['Logging'] = None

    def get_printable_tree(self):
        if False:
            i = 10
            return i + 15
        tree = ET.Element('DistributionConfig')
        tree.attrib['xmlns'] = DistributionConfig.xmlns
        s3org = appendXmlTextNode('S3Origin', '', tree)
        appendXmlTextNode('DNSName', self.info['S3Origin']['DNSName'], s3org)
        appendXmlTextNode('CallerReference', self.info['CallerReference'], tree)
        for cname in self.info['CNAME']:
            appendXmlTextNode('CNAME', cname.lower(), tree)
        if self.info['Comment']:
            appendXmlTextNode('Comment', self.info['Comment'], tree)
        appendXmlTextNode('Enabled', str(self.info['Enabled']).lower(), tree)
        if str(self.info['DefaultRootObject']):
            appendXmlTextNode('DefaultRootObject', str(self.info['DefaultRootObject']), tree)
        if self.info['Logging']:
            logging_el = ET.Element('Logging')
            appendXmlTextNode('Bucket', getHostnameFromBucket(self.info['Logging'].bucket()), logging_el)
            appendXmlTextNode('Prefix', self.info['Logging'].object(), logging_el)
            tree.append(logging_el)
        return tree

    def __unicode__(self):
        if False:
            while True:
                i = 10
        return decode_from_s3(ET.tostring(self.get_printable_tree()))

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if PY3:
            return ET.tostring(self.get_printable_tree(), encoding='unicode')
        else:
            return ET.tostring(self.get_printable_tree())

class Invalidation(object):

    def __init__(self, xml):
        if False:
            while True:
                i = 10
        tree = getTreeFromXml(xml)
        if tree.tag != 'Invalidation':
            raise ValueError('Expected <Invalidation /> xml, got: <%s />' % tree.tag)
        self.parse(tree)

    def parse(self, tree):
        if False:
            return 10
        self.info = getDictFromTree(tree)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.info)

class InvalidationList(object):

    def __init__(self, xml):
        if False:
            print('Hello World!')
        tree = getTreeFromXml(xml)
        if tree.tag != 'InvalidationList':
            raise ValueError('Expected <InvalidationList /> xml, got: <%s />' % tree.tag)
        self.parse(tree)

    def parse(self, tree):
        if False:
            return 10
        self.info = getDictFromTree(tree)

    def __str__(self):
        if False:
            return 10
        return str(self.info)

class InvalidationBatch(object):

    def __init__(self, reference=None, distribution=None, paths=[]):
        if False:
            while True:
                i = 10
        if reference:
            self.reference = reference
        else:
            if not distribution:
                distribution = '0'
            self.reference = '%s.%s.%s' % (distribution, datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'), random.randint(1000, 9999))
        self.paths = []
        self.add_objects(paths)

    def add_objects(self, paths):
        if False:
            while True:
                i = 10
        self.paths.extend(paths)

    def get_reference(self):
        if False:
            print('Hello World!')
        return self.reference

    def get_printable_tree(self):
        if False:
            return 10
        tree = ET.Element('InvalidationBatch')
        for path in self.paths:
            if len(path) < 1 or path[0] != '/':
                path = '/' + path
            appendXmlTextNode('Path', path, tree)
        appendXmlTextNode('CallerReference', self.reference, tree)
        return tree

    def __unicode__(self):
        if False:
            for i in range(10):
                print('nop')
        return decode_from_s3(ET.tostring(self.get_printable_tree()))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if PY3:
            return ET.tostring(self.get_printable_tree(), encoding='unicode')
        else:
            return ET.tostring(self.get_printable_tree())

class CloudFront(object):
    operations = {'CreateDist': {'method': 'POST', 'resource': ''}, 'DeleteDist': {'method': 'DELETE', 'resource': '/%(dist_id)s'}, 'GetList': {'method': 'GET', 'resource': ''}, 'GetDistInfo': {'method': 'GET', 'resource': '/%(dist_id)s'}, 'GetDistConfig': {'method': 'GET', 'resource': '/%(dist_id)s/config'}, 'SetDistConfig': {'method': 'PUT', 'resource': '/%(dist_id)s/config'}, 'Invalidate': {'method': 'POST', 'resource': '/%(dist_id)s/invalidation'}, 'GetInvalList': {'method': 'GET', 'resource': '/%(dist_id)s/invalidation'}, 'GetInvalInfo': {'method': 'GET', 'resource': '/%(dist_id)s/invalidation/%(request_id)s'}}
    dist_list = None

    def __init__(self, config):
        if False:
            print('Hello World!')
        self.config = config

    def GetList(self):
        if False:
            while True:
                i = 10
        response = self.send_request('GetList')
        response['dist_list'] = DistributionList(response['data'])
        if response['dist_list'].info['IsTruncated']:
            raise NotImplementedError('List is truncated. Ask s3cmd author to add support.')
        return response

    def CreateDistribution(self, uri, cnames_add=[], comment=None, logging=None, default_root_object=None):
        if False:
            return 10
        dist_config = DistributionConfig()
        dist_config.info['Enabled'] = True
        dist_config.info['S3Origin']['DNSName'] = uri.host_name()
        dist_config.info['CallerReference'] = str(uri)
        dist_config.info['DefaultRootObject'] = default_root_object
        if comment == None:
            dist_config.info['Comment'] = uri.public_url()
        else:
            dist_config.info['Comment'] = comment
        for cname in cnames_add:
            if dist_config.info['CNAME'].count(cname) == 0:
                dist_config.info['CNAME'].append(cname)
        if logging:
            dist_config.info['Logging'] = S3UriS3(logging)
        request_body = str(dist_config)
        debug('CreateDistribution(): request_body: %s' % request_body)
        response = self.send_request('CreateDist', body=request_body)
        response['distribution'] = Distribution(response['data'])
        return response

    def ModifyDistribution(self, cfuri, cnames_add=[], cnames_remove=[], comment=None, enabled=None, logging=None, default_root_object=None):
        if False:
            for i in range(10):
                print('nop')
        if cfuri.type != 'cf':
            raise ValueError('Expected CFUri instead of: %s' % cfuri)
        info('Checking current status of %s' % cfuri)
        response = self.GetDistConfig(cfuri)
        dc = response['dist_config']
        if enabled != None:
            dc.info['Enabled'] = enabled
        if comment != None:
            dc.info['Comment'] = comment
        if default_root_object != None:
            dc.info['DefaultRootObject'] = default_root_object
        for cname in cnames_add:
            if dc.info['CNAME'].count(cname) == 0:
                dc.info['CNAME'].append(cname)
        for cname in cnames_remove:
            while dc.info['CNAME'].count(cname) > 0:
                dc.info['CNAME'].remove(cname)
        if logging != None:
            if logging == False:
                dc.info['Logging'] = False
            else:
                dc.info['Logging'] = S3UriS3(logging)
        response = self.SetDistConfig(cfuri, dc, response['headers']['etag'])
        return response

    def DeleteDistribution(self, cfuri):
        if False:
            while True:
                i = 10
        if cfuri.type != 'cf':
            raise ValueError('Expected CFUri instead of: %s' % cfuri)
        info('Checking current status of %s' % cfuri)
        response = self.GetDistConfig(cfuri)
        if response['dist_config'].info['Enabled']:
            info('Distribution is ENABLED. Disabling first.')
            response['dist_config'].info['Enabled'] = False
            response = self.SetDistConfig(cfuri, response['dist_config'], response['headers']['etag'])
            warning('Waiting for Distribution to become disabled.')
            warning('This may take several minutes, please wait.')
            while True:
                response = self.GetDistInfo(cfuri)
                d = response['distribution']
                if d.info['Status'] == 'Deployed' and d.info['Enabled'] == False:
                    info('Distribution is now disabled')
                    break
                warning('Still waiting...')
                time.sleep(10)
        headers = SortedDict(ignore_case=True)
        headers['if-match'] = response['headers']['etag']
        response = self.send_request('DeleteDist', dist_id=cfuri.dist_id(), headers=headers)
        return response

    def GetDistInfo(self, cfuri):
        if False:
            return 10
        if cfuri.type != 'cf':
            raise ValueError('Expected CFUri instead of: %s' % cfuri)
        response = self.send_request('GetDistInfo', dist_id=cfuri.dist_id())
        response['distribution'] = Distribution(response['data'])
        return response

    def GetDistConfig(self, cfuri):
        if False:
            i = 10
            return i + 15
        if cfuri.type != 'cf':
            raise ValueError('Expected CFUri instead of: %s' % cfuri)
        response = self.send_request('GetDistConfig', dist_id=cfuri.dist_id())
        response['dist_config'] = DistributionConfig(response['data'])
        return response

    def SetDistConfig(self, cfuri, dist_config, etag=None):
        if False:
            return 10
        if etag == None:
            debug('SetDistConfig(): Etag not set. Fetching it first.')
            etag = self.GetDistConfig(cfuri)['headers']['etag']
        debug('SetDistConfig(): Etag = %s' % etag)
        request_body = str(dist_config)
        debug('SetDistConfig(): request_body: %s' % request_body)
        headers = SortedDict(ignore_case=True)
        headers['if-match'] = etag
        response = self.send_request('SetDistConfig', dist_id=cfuri.dist_id(), body=request_body, headers=headers)
        return response

    def InvalidateObjects(self, uri, paths, default_index_file, invalidate_default_index_on_cf, invalidate_default_index_root_on_cf):
        if False:
            while True:
                i = 10
        if default_index_file is not None and (not invalidate_default_index_on_cf or invalidate_default_index_root_on_cf):
            new_paths = []
            default_index_suffix = '/' + default_index_file
            for path in paths:
                if path.endswith(default_index_suffix) or path == default_index_file:
                    if invalidate_default_index_on_cf:
                        new_paths.append(path)
                    if invalidate_default_index_root_on_cf:
                        new_paths.append(path[:-len(default_index_file)])
                else:
                    new_paths.append(path)
            paths = new_paths
        cfuris = self.get_dist_name_for_bucket(uri)
        if len(paths) > 999:
            try:
                tmp_filename = Utils.mktmpfile()
                with open(deunicodise(tmp_filename), 'w') as fp:
                    fp.write(deunicodise('\n'.join(paths) + '\n'))
                warning('Request to invalidate %d paths (max 999 supported)' % len(paths))
                warning('All the paths are now saved in: %s' % tmp_filename)
            except Exception:
                pass
            raise ParameterError('Too many paths to invalidate')
        responses = []
        for cfuri in cfuris:
            invalbatch = InvalidationBatch(distribution=cfuri.dist_id(), paths=paths)
            debug('InvalidateObjects(): request_body: %s' % invalbatch)
            response = self.send_request('Invalidate', dist_id=cfuri.dist_id(), body=str(invalbatch))
            response['dist_id'] = cfuri.dist_id()
            if response['status'] == 201:
                inval_info = Invalidation(response['data']).info
                response['request_id'] = inval_info['Id']
            debug('InvalidateObjects(): response: %s' % response)
            responses.append(response)
        return responses

    def GetInvalList(self, cfuri):
        if False:
            for i in range(10):
                print('nop')
        if cfuri.type != 'cf':
            raise ValueError('Expected CFUri instead of: %s' % cfuri)
        response = self.send_request('GetInvalList', dist_id=cfuri.dist_id())
        response['inval_list'] = InvalidationList(response['data'])
        return response

    def GetInvalInfo(self, cfuri):
        if False:
            return 10
        if cfuri.type != 'cf':
            raise ValueError('Expected CFUri instead of: %s' % cfuri)
        if cfuri.request_id() is None:
            raise ValueError('Expected CFUri with Request ID')
        response = self.send_request('GetInvalInfo', dist_id=cfuri.dist_id(), request_id=cfuri.request_id())
        response['inval_status'] = Invalidation(response['data'])
        return response

    def send_request(self, op_name, dist_id=None, request_id=None, body=None, headers=None, retries=None):
        if False:
            print('Hello World!')
        if retries is None:
            retries = self.config.max_retries
        if headers is None:
            headers = SortedDict(ignore_case=True)
        operation = self.operations[op_name]
        if body:
            headers['content-type'] = 'text/plain'
        request = self.create_request(operation, dist_id, request_id, headers)
        conn = self.get_connection()
        debug('send_request(): %s %s' % (request['method'], request['resource']))
        conn.c.request(request['method'], request['resource'], body, request['headers'])
        http_response = conn.c.getresponse()
        response = {}
        response['status'] = http_response.status
        response['reason'] = http_response.reason
        response['headers'] = convertHeaderTupleListToDict(http_response.getheaders())
        response['data'] = http_response.read()
        ConnMan.put(conn)
        debug('CloudFront: response: %r' % response)
        if response['status'] >= 500:
            e = CloudFrontError(response)
            if retries:
                warning(u'Retrying failed request: %s (%s)' % (op_name, e))
                warning('Waiting %d sec...' % self._fail_wait(retries))
                time.sleep(self._fail_wait(retries))
                return self.send_request(op_name, dist_id, body=body, retries=retries - 1)
            else:
                raise e
        if response['status'] < 200 or response['status'] > 299:
            raise CloudFrontError(response)
        return response

    def create_request(self, operation, dist_id=None, request_id=None, headers=None):
        if False:
            i = 10
            return i + 15
        resource = cloudfront_resource + operation['resource'] % {'dist_id': dist_id, 'request_id': request_id}
        if not headers:
            headers = SortedDict(ignore_case=True)
        if 'date' in headers:
            if 'x-amz-date' not in headers:
                headers['x-amz-date'] = headers['date']
            del headers['date']
        if 'x-amz-date' not in headers:
            headers['x-amz-date'] = time.strftime('%a, %d %b %Y %H:%M:%S +0000', time.gmtime())
        if len(self.config.access_token) > 0:
            self.config.role_refresh()
            headers['x-amz-security-token'] = self.config.access_token
        signature = self.sign_request(headers)
        headers['Authorization'] = 'AWS ' + self.config.access_key + ':' + signature
        request = {}
        request['resource'] = resource
        request['headers'] = headers
        request['method'] = operation['method']
        return request

    def sign_request(self, headers):
        if False:
            while True:
                i = 10
        string_to_sign = headers['x-amz-date']
        signature = decode_from_s3(sign_string_v2(encode_to_s3(string_to_sign)))
        debug(u"CloudFront.sign_request('%s') = %s" % (string_to_sign, signature))
        return signature

    def get_connection(self):
        if False:
            while True:
                i = 10
        conn = ConnMan.get(self.config.cloudfront_host, ssl=True)
        return conn

    def _fail_wait(self, retries):
        if False:
            return 10
        return (self.config.max_retries - retries + 1) * 3

    def get_dist_name_for_bucket(self, uri):
        if False:
            for i in range(10):
                print('nop')
        if uri.type == 'cf':
            return [uri]
        if uri.type != 's3':
            raise ParameterError('CloudFront or S3 URI required instead of: %s' % uri)
        debug('_get_dist_name_for_bucket(%r)' % uri)
        if CloudFront.dist_list is None:
            response = self.GetList()
            CloudFront.dist_list = {}
            for d in response['dist_list'].dist_summs:
                distListIndex = ''
                if 'S3Origin' in d.info:
                    distListIndex = getBucketFromHostname(d.info['S3Origin']['DNSName'])[0]
                elif 'CustomOrigin' in d.info:
                    distListIndex = getBucketFromHostname(d.info['CustomOrigin']['DNSName'])[0]
                    distListIndex = distListIndex[:len(uri.bucket())]
                else:
                    continue
                if CloudFront.dist_list.get(distListIndex, None) is None:
                    CloudFront.dist_list[distListIndex] = set()
                CloudFront.dist_list[distListIndex].add(d.uri())
            debug('dist_list: %s' % CloudFront.dist_list)
        try:
            return CloudFront.dist_list[uri.bucket()]
        except Exception as e:
            debug(e)
            raise ParameterError('Unable to translate S3 URI to CloudFront distribution name: %s' % uri)

class Cmd(object):
    """
    Class that implements CloudFront commands
    """

    class Options(object):
        cf_cnames_add = []
        cf_cnames_remove = []
        cf_comment = None
        cf_enable = None
        cf_logging = None
        cf_default_root_object = None

        def option_list(self):
            if False:
                print('Hello World!')
            return [opt for opt in dir(self) if opt.startswith('cf_')]

        def update_option(self, option, value):
            if False:
                i = 10
                return i + 15
            setattr(Cmd.options, option, value)
    options = Options()

    @staticmethod
    def _parse_args(args):
        if False:
            print('Hello World!')
        cf = CloudFront(Config())
        cfuris = []
        for arg in args:
            uris = cf.get_dist_name_for_bucket(S3Uri(arg))
            cfuris.extend(uris)
        return cfuris

    @staticmethod
    def info(args):
        if False:
            i = 10
            return i + 15
        cf = CloudFront(Config())
        if not args:
            response = cf.GetList()
            for d in response['dist_list'].dist_summs:
                if 'S3Origin' in d.info:
                    origin = S3UriS3.httpurl_to_s3uri(d.info['S3Origin']['DNSName'])
                elif 'CustomOrigin' in d.info:
                    origin = 'http://%s/' % d.info['CustomOrigin']['DNSName']
                else:
                    origin = '<unknown>'
                pretty_output('Origin', origin)
                pretty_output('DistId', d.uri())
                pretty_output('DomainName', d.info['DomainName'])
                if 'CNAME' in d.info:
                    pretty_output('CNAMEs', ', '.join(d.info['CNAME']))
                pretty_output('Status', d.info['Status'])
                pretty_output('Enabled', d.info['Enabled'])
                output('')
        else:
            cfuris = Cmd._parse_args(args)
            for cfuri in cfuris:
                response = cf.GetDistInfo(cfuri)
                d = response['distribution']
                dc = d.info['DistributionConfig']
                if 'S3Origin' in dc.info:
                    origin = S3UriS3.httpurl_to_s3uri(dc.info['S3Origin']['DNSName'])
                elif 'CustomOrigin' in dc.info:
                    origin = 'http://%s/' % dc.info['CustomOrigin']['DNSName']
                else:
                    origin = '<unknown>'
                pretty_output('Origin', origin)
                pretty_output('DistId', d.uri())
                pretty_output('DomainName', d.info['DomainName'])
                if 'CNAME' in dc.info:
                    pretty_output('CNAMEs', ', '.join(dc.info['CNAME']))
                pretty_output('Status', d.info['Status'])
                pretty_output('Comment', dc.info['Comment'])
                pretty_output('Enabled', dc.info['Enabled'])
                pretty_output('DfltRootObject', dc.info['DefaultRootObject'])
                pretty_output('Logging', dc.info['Logging'] or 'Disabled')
                pretty_output('Etag', response['headers']['etag'])

    @staticmethod
    def create(args):
        if False:
            i = 10
            return i + 15
        cf = CloudFront(Config())
        buckets = []
        for arg in args:
            uri = S3Uri(arg)
            if uri.type != 's3':
                raise ParameterError('Distribution can only be created from a s3:// URI instead of: %s' % arg)
            if uri.object():
                raise ParameterError('Use s3:// URI with a bucket name only instead of: %s' % arg)
            if not uri.is_dns_compatible():
                raise ParameterError('CloudFront can only handle lowercase-named buckets.')
            buckets.append(uri)
        if not buckets:
            raise ParameterError('No valid bucket names found')
        for uri in buckets:
            info('Creating distribution from: %s' % uri)
            response = cf.CreateDistribution(uri, cnames_add=Cmd.options.cf_cnames_add, comment=Cmd.options.cf_comment, logging=Cmd.options.cf_logging, default_root_object=Cmd.options.cf_default_root_object)
            d = response['distribution']
            dc = d.info['DistributionConfig']
            output('Distribution created:')
            pretty_output('Origin', S3UriS3.httpurl_to_s3uri(dc.info['S3Origin']['DNSName']))
            pretty_output('DistId', d.uri())
            pretty_output('DomainName', d.info['DomainName'])
            pretty_output('CNAMEs', ', '.join(dc.info['CNAME']))
            pretty_output('Comment', dc.info['Comment'])
            pretty_output('Status', d.info['Status'])
            pretty_output('Enabled', dc.info['Enabled'])
            pretty_output('DefaultRootObject', dc.info['DefaultRootObject'])
            pretty_output('Etag', response['headers']['etag'])

    @staticmethod
    def delete(args):
        if False:
            print('Hello World!')
        cf = CloudFront(Config())
        cfuris = Cmd._parse_args(args)
        for cfuri in cfuris:
            response = cf.DeleteDistribution(cfuri)
            if response['status'] >= 400:
                error('Distribution %s could not be deleted: %s' % (cfuri, response['reason']))
            output('Distribution %s deleted' % cfuri)

    @staticmethod
    def modify(args):
        if False:
            while True:
                i = 10
        cf = CloudFront(Config())
        if len(args) > 1:
            raise ParameterError('Too many parameters. Modify one Distribution at a time.')
        try:
            cfuri = Cmd._parse_args(args)[0]
        except IndexError:
            raise ParameterError('No valid Distribution URI found.')
        response = cf.ModifyDistribution(cfuri, cnames_add=Cmd.options.cf_cnames_add, cnames_remove=Cmd.options.cf_cnames_remove, comment=Cmd.options.cf_comment, enabled=Cmd.options.cf_enable, logging=Cmd.options.cf_logging, default_root_object=Cmd.options.cf_default_root_object)
        if response['status'] >= 400:
            error('Distribution %s could not be modified: %s' % (cfuri, response['reason']))
        output('Distribution modified: %s' % cfuri)
        response = cf.GetDistInfo(cfuri)
        d = response['distribution']
        dc = d.info['DistributionConfig']
        pretty_output('Origin', S3UriS3.httpurl_to_s3uri(dc.info['S3Origin']['DNSName']))
        pretty_output('DistId', d.uri())
        pretty_output('DomainName', d.info['DomainName'])
        pretty_output('Status', d.info['Status'])
        pretty_output('CNAMEs', ', '.join(dc.info['CNAME']))
        pretty_output('Comment', dc.info['Comment'])
        pretty_output('Enabled', dc.info['Enabled'])
        pretty_output('DefaultRootObject', dc.info['DefaultRootObject'])
        pretty_output('Etag', response['headers']['etag'])

    @staticmethod
    def invalinfo(args):
        if False:
            return 10
        cf = CloudFront(Config())
        cfuris = Cmd._parse_args(args)
        requests = []
        for cfuri in cfuris:
            if cfuri.request_id():
                requests.append(str(cfuri))
            else:
                inval_list = cf.GetInvalList(cfuri)
                try:
                    for i in inval_list['inval_list'].info['InvalidationSummary']:
                        requests.append('/'.join(['cf:/', cfuri.dist_id(), i['Id']]))
                except Exception:
                    continue
        for req in requests:
            cfuri = S3Uri(req)
            inval_info = cf.GetInvalInfo(cfuri)
            st = inval_info['inval_status'].info
            paths = st['InvalidationBatch']['Path']
            nr_of_paths = len(paths) if isinstance(paths, list) else 1
            pretty_output('URI', str(cfuri))
            pretty_output('Status', st['Status'])
            pretty_output('Created', st['CreateTime'])
            pretty_output('Nr of paths', nr_of_paths)
            pretty_output('Reference', st['InvalidationBatch']['CallerReference'])
            output('')

    @staticmethod
    def invalidate(args):
        if False:
            return 10
        cfg = Config()
        cf = CloudFront(cfg)
        s3 = S3(cfg)
        bucket_paths = defaultdict(list)
        for arg in args:
            uri = S3Uri(arg)
            uobject = uri.object()
            if not uobject:
                uobject = '*'
            elif uobject[-1] == '/':
                uobject += '*'
            bucket_paths[uri.bucket()].append(uobject)
        ret = EX_OK
        params = []
        for (bucket, paths) in bucket_paths.items():
            base_uri = S3Uri(u's3://%s' % bucket)
            cfuri = next(iter(cf.get_dist_name_for_bucket(base_uri)))
            default_index_file = None
            if cfg.invalidate_default_index_on_cf or cfg.invalidate_default_index_root_on_cf:
                info_response = s3.website_info(base_uri, cfg.bucket_location)
                if info_response:
                    default_index_file = info_response['index_document']
                    if not default_index_file:
                        default_index_file = None
            if cfg.dry_run:
                fulluri_paths = [S3UriS3.compose_uri(bucket, path) for path in paths]
                output(u'[--dry-run] Would invalidate %r' % fulluri_paths)
                continue
            params.append((bucket, paths, base_uri, cfuri, default_index_file))
        if cfg.dry_run:
            warning(u'Exiting now because of --dry-run')
            return EX_OK
        nb_success = 0
        first = True
        for (bucket, paths, base_uri, cfuri, default_index_file) in params:
            if not first:
                output('')
            else:
                first = False
            results = cf.InvalidateObjects(cfuri, paths, default_index_file, cfg.invalidate_default_index_on_cf, cfg.invalidate_default_index_root_on_cf)
            dist_id = cfuri.dist_id()
            pretty_output('URI', str(base_uri))
            pretty_output('DistId', dist_id)
            pretty_output('Nr of paths', len(paths))
            for result in results:
                result_code = result['status']
                if result_code != 201:
                    pretty_output('Status', 'Failed: %d' % result_code)
                    ret = EX_GENERAL
                    continue
                request_id = result['request_id']
                nb_success += 1
                pretty_output('Status', 'Created')
                pretty_output('RequestId', request_id)
                pretty_output('Info', u'Check progress with: s3cmd cfinvalinfo %s/%s' % (dist_id, request_id))
            if ret != EX_OK and cfg.stop_on_error:
                error(u'Exiting now because of --stop-on-error')
                break
        if ret != EX_OK and nb_success:
            ret = EX_PARTIAL
        return ret