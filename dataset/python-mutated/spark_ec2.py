from __future__ import division, print_function, with_statement
import codecs
import hashlib
import itertools
import logging
import os
import os.path
import pipes
import random
import shutil
import string
from stat import S_IRUSR
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import warnings
from datetime import datetime
from optparse import OptionParser
from sys import stderr
if sys.version < '3':
    from urllib2 import urlopen, Request, HTTPError
else:
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    raw_input = input
    xrange = range
SPARK_EC2_VERSION = '1.6.0'
SPARK_EC2_DIR = os.path.dirname(os.path.realpath(__file__))
VALID_SPARK_VERSIONS = set(['0.7.3', '0.8.0', '0.8.1', '0.9.0', '0.9.1', '0.9.2', '1.0.0', '1.0.1', '1.0.2', '1.1.0', '1.1.1', '1.2.0', '1.2.1', '1.3.0', '1.3.1', '1.4.0', '1.4.1', '1.5.0', '1.5.1', '1.5.2', '1.6.0'])
SPARK_TACHYON_MAP = {'1.0.0': '0.4.1', '1.0.1': '0.4.1', '1.0.2': '0.4.1', '1.1.0': '0.5.0', '1.1.1': '0.5.0', '1.2.0': '0.5.0', '1.2.1': '0.5.0', '1.3.0': '0.5.0', '1.3.1': '0.5.0', '1.4.0': '0.6.4', '1.4.1': '0.6.4', '1.5.0': '0.7.1', '1.5.1': '0.7.1', '1.5.2': '0.7.1', '1.6.0': '0.8.2'}
DEFAULT_SPARK_VERSION = SPARK_EC2_VERSION
DEFAULT_SPARK_GITHUB_REPO = 'https://github.com/apache/spark'
DEFAULT_SPARK_EC2_GITHUB_REPO = 'https://github.com/anfeng/spark-ec2'
DEFAULT_SPARK_EC2_BRANCH = 'branch-2.6'

def setup_external_libs(libs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Download external libraries from PyPI to SPARK_EC2_DIR/lib/ and prepend them to our PATH.\n    '
    PYPI_URL_PREFIX = 'https://pypi.python.org/packages/source'
    SPARK_EC2_LIB_DIR = os.path.join(SPARK_EC2_DIR, 'lib')
    if not os.path.exists(SPARK_EC2_LIB_DIR):
        print('Downloading external libraries that spark-ec2 needs from PyPI to {path}...'.format(path=SPARK_EC2_LIB_DIR))
        print('This should be a one-time operation.')
        os.mkdir(SPARK_EC2_LIB_DIR)
    for lib in libs:
        versioned_lib_name = '{n}-{v}'.format(n=lib['name'], v=lib['version'])
        lib_dir = os.path.join(SPARK_EC2_LIB_DIR, versioned_lib_name)
        if not os.path.isdir(lib_dir):
            tgz_file_path = os.path.join(SPARK_EC2_LIB_DIR, versioned_lib_name + '.tar.gz')
            print(' - Downloading {lib}...'.format(lib=lib['name']))
            download_stream = urlopen('{prefix}/{first_letter}/{lib_name}/{lib_name}-{lib_version}.tar.gz'.format(prefix=PYPI_URL_PREFIX, first_letter=lib['name'][:1], lib_name=lib['name'], lib_version=lib['version']))
            with open(tgz_file_path, 'wb') as tgz_file:
                tgz_file.write(download_stream.read())
            with open(tgz_file_path, 'rb') as tar:
                if hashlib.md5(tar.read()).hexdigest() != lib['md5']:
                    print('ERROR: Got wrong md5sum for {lib}.'.format(lib=lib['name']), file=stderr)
                    sys.exit(1)
            tar = tarfile.open(tgz_file_path)
            tar.extractall(path=SPARK_EC2_LIB_DIR)
            tar.close()
            os.remove(tgz_file_path)
            print(' - Finished downloading {lib}.'.format(lib=lib['name']))
        sys.path.insert(1, lib_dir)
external_libs = [{'name': 'boto', 'version': '2.34.0', 'md5': '5556223d2d0cc4d06dd4829e671dcecd'}]
setup_external_libs(external_libs)
import boto
from boto.ec2.blockdevicemapping import BlockDeviceMapping, BlockDeviceType, EBSBlockDeviceType
from boto import ec2

class UsageError(Exception):
    pass

def parse_args():
    if False:
        print('Hello World!')
    parser = OptionParser(prog='spark-ec2', version='%prog {v}'.format(v=SPARK_EC2_VERSION), usage='%prog [options] <action> <cluster_name>\n\n' + '<action> can be: launch, destroy, login, stop, start, get-master, reboot-slaves')
    parser.add_option('-s', '--slaves', type='int', default=1, help='Number of slaves to launch (default: %default)')
    parser.add_option('-w', '--wait', type='int', help='DEPRECATED (no longer necessary) - Seconds to wait for nodes to start')
    parser.add_option('-k', '--key-pair', help='Key pair to use on instances')
    parser.add_option('-i', '--identity-file', help='SSH private key file to use for logging into instances')
    parser.add_option('-p', '--profile', default=None, help='If you have multiple profiles (AWS or boto config), you can configure ' + 'additional, named profiles by using this option (default: %default)')
    parser.add_option('-t', '--instance-type', default='m1.large', help='Type of instance to launch (default: %default). ' + "WARNING: must be 64-bit; small instances won't work")
    parser.add_option('-m', '--master-instance-type', default='', help='Master instance type (leave empty for same as instance-type)')
    parser.add_option('-r', '--region', default='us-east-1', help='EC2 region used to launch instances in, or to find them in (default: %default)')
    parser.add_option('-z', '--zone', default='', help="Availability zone to launch instances in, or 'all' to spread " + 'slaves across multiple (an additional $0.01/Gb for bandwidth' + 'between zones applies) (default: a single zone chosen at random)')
    parser.add_option('-a', '--ami', help='Amazon Machine Image ID to use')
    parser.add_option('-v', '--spark-version', default=DEFAULT_SPARK_VERSION, help="Version of Spark to use: 'X.Y.Z' or a specific git hash (default: %default)")
    parser.add_option('--spark-git-repo', default=DEFAULT_SPARK_GITHUB_REPO, help='Github repo from which to checkout supplied commit hash (default: %default)')
    parser.add_option('--spark-ec2-git-repo', default=DEFAULT_SPARK_EC2_GITHUB_REPO, help='Github repo from which to checkout spark-ec2 (default: %default)')
    parser.add_option('--spark-ec2-git-branch', default=DEFAULT_SPARK_EC2_BRANCH, help='Github repo branch of spark-ec2 to use (default: %default)')
    parser.add_option('--deploy-root-dir', default=None, help='A directory to copy into / on the first master. ' + 'Must be absolute. Note that a trailing slash is handled as per rsync: ' + 'If you omit it, the last directory of the --deploy-root-dir path will be created ' + 'in / before copying its contents. If you append the trailing slash, ' + 'the directory is not created and its contents are copied directly into /. ' + '(default: %default).')
    parser.add_option('--hadoop-major-version', default='1', help='Major version of Hadoop. Valid options are 1 (Hadoop 1.0.4), 2 (CDH 4.2.0), yarn ' + '(Hadoop 2.4.0) (default: %default)')
    parser.add_option('-D', metavar='[ADDRESS:]PORT', dest='proxy_port', help='Use SSH dynamic port forwarding to create a SOCKS proxy at ' + 'the given local address (for use with login)')
    parser.add_option('--resume', action='store_true', default=False, help='Resume installation on a previously launched cluster ' + '(for debugging)')
    parser.add_option('--ebs-vol-size', metavar='SIZE', type='int', default=0, help='Size (in GB) of each EBS volume.')
    parser.add_option('--ebs-vol-type', default='standard', help="EBS volume type (e.g. 'gp2', 'standard').")
    parser.add_option('--ebs-vol-num', type='int', default=1, help='Number of EBS volumes to attach to each node as /vol[x]. ' + 'The volumes will be deleted when the instances terminate. ' + 'Only possible on EBS-backed AMIs. ' + 'EBS volumes are only attached if --ebs-vol-size > 0. ' + 'Only support up to 8 EBS volumes.')
    parser.add_option('--placement-group', type='string', default=None, help='Which placement group to try and launch ' + 'instances into. Assumes placement group is already ' + 'created.')
    parser.add_option('--swap', metavar='SWAP', type='int', default=1024, help='Swap space to set up per node, in MB (default: %default)')
    parser.add_option('--spot-price', metavar='PRICE', type='float', help='If specified, launch slaves as spot instances with the given ' + 'maximum price (in dollars)')
    parser.add_option('--ganglia', action='store_true', default=True, help='Setup Ganglia monitoring on cluster (default: %default). NOTE: ' + 'the Ganglia page will be publicly accessible')
    parser.add_option('--no-ganglia', action='store_false', dest='ganglia', help='Disable Ganglia monitoring for the cluster')
    parser.add_option('-u', '--user', default='root', help='The SSH user you want to connect as (default: %default)')
    parser.add_option('--delete-groups', action='store_true', default=False, help='When destroying a cluster, delete the security groups that were created')
    parser.add_option('--use-existing-master', action='store_true', default=False, help='Launch fresh slaves, but use an existing stopped master if possible')
    parser.add_option('--worker-instances', type='int', default=1, help='Number of instances per worker: variable SPARK_WORKER_INSTANCES. Not used if YARN ' + 'is used as Hadoop major version (default: %default)')
    parser.add_option('--master-opts', type='string', default='', help='Extra options to give to master through SPARK_MASTER_OPTS variable ' + '(e.g -Dspark.worker.timeout=180)')
    parser.add_option('--user-data', type='string', default='', help='Path to a user-data file (most AMIs interpret this as an initialization script)')
    parser.add_option('--authorized-address', type='string', default='0.0.0.0/0', help='Address to authorize on created security groups (default: %default)')
    parser.add_option('--additional-security-group', type='string', default='', help='Additional security group to place the machines in')
    parser.add_option('--additional-tags', type='string', default='', help='Additional tags to set on the machines; tags are comma-separated, while name and ' + 'value are colon separated; ex: "Task:MySparkProject,Env:production"')
    parser.add_option('--copy-aws-credentials', action='store_true', default=False, help='Add AWS credentials to hadoop configuration to allow Spark to access S3')
    parser.add_option('--subnet-id', default=None, help='VPC subnet to launch instances in')
    parser.add_option('--vpc-id', default=None, help='VPC to launch instances in')
    parser.add_option('--private-ips', action='store_true', default=False, help='Use private IPs for instances rather than public if VPC/subnet ' + 'requires that.')
    parser.add_option('--instance-initiated-shutdown-behavior', default='stop', choices=['stop', 'terminate'], help='Whether instances should terminate when shut down or just stop')
    parser.add_option('--instance-profile-name', default=None, help='IAM profile name to launch instances under')
    (opts, args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(1)
    (action, cluster_name) = args
    home_dir = os.getenv('HOME')
    if home_dir is None or not os.path.isfile(home_dir + '/.boto'):
        if not os.path.isfile('/etc/boto.cfg'):
            if not os.path.isfile(home_dir + '/.aws/credentials'):
                if os.getenv('AWS_ACCESS_KEY_ID') is None:
                    print('ERROR: The environment variable AWS_ACCESS_KEY_ID must be set', file=stderr)
                    sys.exit(1)
                if os.getenv('AWS_SECRET_ACCESS_KEY') is None:
                    print('ERROR: The environment variable AWS_SECRET_ACCESS_KEY must be set', file=stderr)
                    sys.exit(1)
    return (opts, action, cluster_name)

def get_or_make_group(conn, name, vpc_id):
    if False:
        for i in range(10):
            print('nop')
    groups = conn.get_all_security_groups()
    group = [g for g in groups if g.name == name]
    if len(group) > 0:
        return group[0]
    else:
        print('Creating security group ' + name)
        return conn.create_security_group(name, 'Spark EC2 group', vpc_id)

def get_validate_spark_version(version, repo):
    if False:
        return 10
    if '.' in version:
        version = version.replace('v', '')
        if version not in VALID_SPARK_VERSIONS:
            print("Don't know about Spark version: {v}".format(v=version), file=stderr)
            sys.exit(1)
        return version
    else:
        github_commit_url = '{repo}/commit/{commit_hash}'.format(repo=repo, commit_hash=version)
        request = Request(github_commit_url)
        request.get_method = lambda : 'HEAD'
        try:
            response = urlopen(request)
        except HTTPError as e:
            print("Couldn't validate Spark commit: {url}".format(url=github_commit_url), file=stderr)
            print('Received HTTP response code of {code}.'.format(code=e.code), file=stderr)
            sys.exit(1)
        return version
EC2_INSTANCE_TYPES = {'c1.medium': 'pvm', 'c1.xlarge': 'pvm', 'c3.large': 'pvm', 'c3.xlarge': 'pvm', 'c3.2xlarge': 'pvm', 'c3.4xlarge': 'pvm', 'c3.8xlarge': 'pvm', 'c4.large': 'hvm', 'c4.xlarge': 'hvm', 'c4.2xlarge': 'hvm', 'c4.4xlarge': 'hvm', 'c4.8xlarge': 'hvm', 'cc1.4xlarge': 'hvm', 'cc2.8xlarge': 'hvm', 'cg1.4xlarge': 'hvm', 'cr1.8xlarge': 'hvm', 'd2.xlarge': 'hvm', 'd2.2xlarge': 'hvm', 'd2.4xlarge': 'hvm', 'd2.8xlarge': 'hvm', 'g2.2xlarge': 'hvm', 'g2.8xlarge': 'hvm', 'hi1.4xlarge': 'pvm', 'hs1.8xlarge': 'pvm', 'i2.xlarge': 'hvm', 'i2.2xlarge': 'hvm', 'i2.4xlarge': 'hvm', 'i2.8xlarge': 'hvm', 'm1.small': 'pvm', 'm1.medium': 'pvm', 'm1.large': 'pvm', 'm1.xlarge': 'pvm', 'm2.xlarge': 'pvm', 'm2.2xlarge': 'pvm', 'm2.4xlarge': 'pvm', 'm3.medium': 'hvm', 'm3.large': 'hvm', 'm3.xlarge': 'hvm', 'm3.2xlarge': 'hvm', 'm4.large': 'hvm', 'm4.xlarge': 'hvm', 'm4.2xlarge': 'hvm', 'm4.4xlarge': 'hvm', 'm4.10xlarge': 'hvm', 'p2.large': 'hvm', 'p2.8xlarge': 'hvm', 'p2.16xlarge': 'hvm', 'r3.large': 'hvm', 'r3.xlarge': 'hvm', 'r3.2xlarge': 'hvm', 'r3.4xlarge': 'hvm', 'r3.8xlarge': 'hvm', 't1.micro': 'pvm', 't2.micro': 'hvm', 't2.small': 'hvm', 't2.medium': 'hvm', 't2.large': 'hvm'}

def get_tachyon_version(spark_version):
    if False:
        return 10
    return SPARK_TACHYON_MAP.get(spark_version, '')

def get_spark_ami(opts):
    if False:
        print('Hello World!')
    if opts.instance_type in EC2_INSTANCE_TYPES:
        instance_type = EC2_INSTANCE_TYPES[opts.instance_type]
    else:
        instance_type = 'pvm'
        print("Don't recognize %s, assuming type is pvm" % opts.instance_type, file=stderr)
    ami_prefix = '{r}/{b}/ami-list'.format(r=opts.spark_ec2_git_repo.replace('https://github.com', 'https://raw.github.com', 1), b=opts.spark_ec2_git_branch)
    ami_path = '%s/%s/%s' % (ami_prefix, opts.region, instance_type)
    reader = codecs.getreader('ascii')
    try:
        ami = reader(urlopen(ami_path)).read().strip()
    except:
        print('Could not resolve AMI at: ' + ami_path, file=stderr)
        sys.exit(1)
    print('Spark AMI: ' + ami)
    return ami

def launch_cluster(conn, opts, cluster_name):
    if False:
        i = 10
        return i + 15
    if opts.identity_file is None:
        print('ERROR: Must provide an identity file (-i) for ssh connections.', file=stderr)
        sys.exit(1)
    if opts.key_pair is None:
        print('ERROR: Must provide a key pair name (-k) to use on instances.', file=stderr)
        sys.exit(1)
    user_data_content = None
    if opts.user_data:
        with open(opts.user_data) as user_data_file:
            user_data_content = user_data_file.read()
    print('Setting up security groups...')
    master_group = get_or_make_group(conn, cluster_name + '-master', opts.vpc_id)
    slave_group = get_or_make_group(conn, cluster_name + '-slaves', opts.vpc_id)
    authorized_address = opts.authorized_address
    if master_group.rules == []:
        if opts.vpc_id is None:
            master_group.authorize(src_group=master_group)
            master_group.authorize(src_group=slave_group)
        else:
            master_group.authorize(ip_protocol='icmp', from_port=-1, to_port=-1, src_group=master_group)
            master_group.authorize(ip_protocol='tcp', from_port=0, to_port=65535, src_group=master_group)
            master_group.authorize(ip_protocol='udp', from_port=0, to_port=65535, src_group=master_group)
            master_group.authorize(ip_protocol='icmp', from_port=-1, to_port=-1, src_group=slave_group)
            master_group.authorize(ip_protocol='tcp', from_port=0, to_port=65535, src_group=slave_group)
            master_group.authorize(ip_protocol='udp', from_port=0, to_port=65535, src_group=slave_group)
        master_group.authorize('tcp', 22, 22, authorized_address)
        master_group.authorize('tcp', 8080, 8081, authorized_address)
        master_group.authorize('tcp', 18080, 18080, authorized_address)
        master_group.authorize('tcp', 19999, 19999, authorized_address)
        master_group.authorize('tcp', 50030, 50030, authorized_address)
        master_group.authorize('tcp', 50070, 50070, authorized_address)
        master_group.authorize('tcp', 60070, 60070, authorized_address)
        master_group.authorize('tcp', 4040, 4045, authorized_address)
        master_group.authorize('tcp', 111, 111, authorized_address)
        master_group.authorize('udp', 111, 111, authorized_address)
        master_group.authorize('tcp', 2049, 2049, authorized_address)
        master_group.authorize('udp', 2049, 2049, authorized_address)
        master_group.authorize('tcp', 4242, 4242, authorized_address)
        master_group.authorize('udp', 4242, 4242, authorized_address)
        master_group.authorize('tcp', 8088, 8088, authorized_address)
        if opts.ganglia:
            master_group.authorize('tcp', 5080, 5080, authorized_address)
    if slave_group.rules == []:
        if opts.vpc_id is None:
            slave_group.authorize(src_group=master_group)
            slave_group.authorize(src_group=slave_group)
        else:
            slave_group.authorize(ip_protocol='icmp', from_port=-1, to_port=-1, src_group=master_group)
            slave_group.authorize(ip_protocol='tcp', from_port=0, to_port=65535, src_group=master_group)
            slave_group.authorize(ip_protocol='udp', from_port=0, to_port=65535, src_group=master_group)
            slave_group.authorize(ip_protocol='icmp', from_port=-1, to_port=-1, src_group=slave_group)
            slave_group.authorize(ip_protocol='tcp', from_port=0, to_port=65535, src_group=slave_group)
            slave_group.authorize(ip_protocol='udp', from_port=0, to_port=65535, src_group=slave_group)
        slave_group.authorize('tcp', 22, 22, authorized_address)
        slave_group.authorize('tcp', 8080, 8081, authorized_address)
        slave_group.authorize('tcp', 50060, 50060, authorized_address)
        slave_group.authorize('tcp', 50075, 50075, authorized_address)
        slave_group.authorize('tcp', 60060, 60060, authorized_address)
        slave_group.authorize('tcp', 60075, 60075, authorized_address)
    (existing_masters, existing_slaves) = get_existing_cluster(conn, opts, cluster_name, die_on_error=False)
    if existing_slaves or (existing_masters and (not opts.use_existing_master)):
        print('ERROR: There are already instances running in group %s or %s' % (master_group.name, slave_group.name), file=stderr)
        sys.exit(1)
    if opts.ami is None:
        opts.ami = get_spark_ami(opts)
    additional_group_ids = []
    if opts.additional_security_group:
        additional_group_ids = [sg.id for sg in conn.get_all_security_groups() if opts.additional_security_group in (sg.name, sg.id)]
    print('Launching instances...')
    try:
        image = conn.get_all_images(image_ids=[opts.ami])[0]
    except:
        print('Could not find AMI ' + opts.ami, file=stderr)
        sys.exit(1)
    block_map = BlockDeviceMapping()
    if opts.ebs_vol_size > 0:
        for i in range(opts.ebs_vol_num):
            device = EBSBlockDeviceType()
            device.size = opts.ebs_vol_size
            device.volume_type = opts.ebs_vol_type
            device.delete_on_termination = True
            block_map['/dev/sd' + chr(ord('s') + i)] = device
    if opts.instance_type.startswith('m3.'):
        for i in range(get_num_disks(opts.instance_type)):
            dev = BlockDeviceType()
            dev.ephemeral_name = 'ephemeral%d' % i
            name = '/dev/sd' + string.ascii_letters[i + 1]
            block_map[name] = dev
    if opts.spot_price is not None:
        print('Requesting %d slaves as spot instances with price $%.3f' % (opts.slaves, opts.spot_price))
        zones = get_zones(conn, opts)
        num_zones = len(zones)
        i = 0
        my_req_ids = []
        for zone in zones:
            num_slaves_this_zone = get_partition(opts.slaves, num_zones, i)
            slave_reqs = conn.request_spot_instances(price=opts.spot_price, image_id=opts.ami, launch_group='launch-group-%s' % cluster_name, placement=zone, count=num_slaves_this_zone, key_name=opts.key_pair, security_group_ids=[slave_group.id] + additional_group_ids, instance_type=opts.instance_type, block_device_map=block_map, subnet_id=opts.subnet_id, placement_group=opts.placement_group, user_data=user_data_content, instance_profile_name=opts.instance_profile_name)
            my_req_ids += [req.id for req in slave_reqs]
            i += 1
        print('Waiting for spot instances to be granted...')
        try:
            while True:
                time.sleep(10)
                reqs = conn.get_all_spot_instance_requests()
                id_to_req = {}
                for r in reqs:
                    id_to_req[r.id] = r
                active_instance_ids = []
                for i in my_req_ids:
                    if i in id_to_req and id_to_req[i].state == 'active':
                        active_instance_ids.append(id_to_req[i].instance_id)
                if len(active_instance_ids) == opts.slaves:
                    print('All %d slaves granted' % opts.slaves)
                    reservations = conn.get_all_reservations(active_instance_ids)
                    slave_nodes = []
                    for r in reservations:
                        slave_nodes += r.instances
                    break
                else:
                    print('%d of %d slaves granted, waiting longer' % (len(active_instance_ids), opts.slaves))
        except:
            print('Canceling spot instance requests')
            conn.cancel_spot_instance_requests(my_req_ids)
            (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name, die_on_error=False)
            running = len(master_nodes) + len(slave_nodes)
            if running:
                print('WARNING: %d instances are still running' % running, file=stderr)
            sys.exit(0)
    else:
        zones = get_zones(conn, opts)
        num_zones = len(zones)
        i = 0
        slave_nodes = []
        for zone in zones:
            num_slaves_this_zone = get_partition(opts.slaves, num_zones, i)
            if num_slaves_this_zone > 0:
                slave_res = image.run(key_name=opts.key_pair, security_group_ids=[slave_group.id] + additional_group_ids, instance_type=opts.instance_type, placement=zone, min_count=num_slaves_this_zone, max_count=num_slaves_this_zone, block_device_map=block_map, subnet_id=opts.subnet_id, placement_group=opts.placement_group, user_data=user_data_content, instance_initiated_shutdown_behavior=opts.instance_initiated_shutdown_behavior, instance_profile_name=opts.instance_profile_name)
                slave_nodes += slave_res.instances
                print('Launched {s} slave{plural_s} in {z}, regid = {r}'.format(s=num_slaves_this_zone, plural_s='' if num_slaves_this_zone == 1 else 's', z=zone, r=slave_res.id))
            i += 1
    if existing_masters:
        print('Starting master...')
        for inst in existing_masters:
            if inst.state not in ['shutting-down', 'terminated']:
                inst.start()
        master_nodes = existing_masters
    else:
        master_type = opts.master_instance_type
        if master_type == '':
            master_type = opts.instance_type
        if opts.zone == 'all':
            opts.zone = random.choice(conn.get_all_zones()).name
        master_res = image.run(key_name=opts.key_pair, security_group_ids=[master_group.id] + additional_group_ids, instance_type=master_type, placement=opts.zone, min_count=1, max_count=1, block_device_map=block_map, subnet_id=opts.subnet_id, placement_group=opts.placement_group, user_data=user_data_content, instance_initiated_shutdown_behavior=opts.instance_initiated_shutdown_behavior, instance_profile_name=opts.instance_profile_name)
        master_nodes = master_res.instances
        print('Launched master in %s, regid = %s' % (zone, master_res.id))
    print('Waiting for AWS to propagate instance metadata...')
    time.sleep(15)
    additional_tags = {}
    if opts.additional_tags.strip():
        additional_tags = dict((map(str.strip, tag.split(':', 1)) for tag in opts.additional_tags.split(',')))
    for master in master_nodes:
        master.add_tags(dict(additional_tags, Name='{cn}-master-{iid}'.format(cn=cluster_name, iid=master.id)))
    for slave in slave_nodes:
        slave.add_tags(dict(additional_tags, Name='{cn}-slave-{iid}'.format(cn=cluster_name, iid=slave.id)))
    return (master_nodes, slave_nodes)

def get_existing_cluster(conn, opts, cluster_name, die_on_error=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the EC2 instances in an existing cluster if available.\n    Returns a tuple of lists of EC2 instance objects for the masters and slaves.\n    '
    print('Searching for existing cluster {c} in region {r}...'.format(c=cluster_name, r=opts.region))

    def get_instances(group_names):
        if False:
            i = 10
            return i + 15
        '\n        Get all non-terminated instances that belong to any of the provided security groups.\n\n        EC2 reservation filters and instance states are documented here:\n            http://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instances.html#options\n        '
        reservations = conn.get_all_reservations(filters={'instance.group-name': group_names})
        instances = itertools.chain.from_iterable((r.instances for r in reservations))
        return [i for i in instances if i.state not in ['shutting-down', 'terminated']]
    master_instances = get_instances([cluster_name + '-master'])
    slave_instances = get_instances([cluster_name + '-slaves'])
    if any((master_instances, slave_instances)):
        print('Found {m} master{plural_m}, {s} slave{plural_s}.'.format(m=len(master_instances), plural_m='' if len(master_instances) == 1 else 's', s=len(slave_instances), plural_s='' if len(slave_instances) == 1 else 's'))
    if not master_instances and die_on_error:
        print('ERROR: Could not find a master for cluster {c} in region {r}.'.format(c=cluster_name, r=opts.region), file=sys.stderr)
        sys.exit(1)
    return (master_instances, slave_instances)

def ssh_cluster(master_nodes, slave_nodes, opts, cmd):
    if False:
        return 10
    master = get_dns_name(master_nodes[0], opts.private_ips)
    ssh(master, opts, cmd)
    for slave in slave_nodes:
        slave_address = get_dns_name(slave, opts.private_ips)
        ssh(slave_address, opts, cmd)

def setup_cluster(conn, master_nodes, slave_nodes, opts, deploy_ssh_key):
    if False:
        print('Hello World!')
    master = get_dns_name(master_nodes[0], opts.private_ips)
    if deploy_ssh_key:
        print("Generating cluster's SSH key on master...")
        key_setup = "\n          [ -f ~/.ssh/id_rsa ] ||\n            (ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa &&\n             cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys)\n        "
        ssh(master, opts, key_setup)
        dot_ssh_tar = ssh_read(master, opts, ['tar', 'c', '.ssh'])
        print("Transferring cluster's SSH key to slaves...")
        for slave in slave_nodes:
            slave_address = get_dns_name(slave, opts.private_ips)
            print(slave_address)
            ssh_write(slave_address, opts, ['tar', 'x'], dot_ssh_tar)
    modules = ['spark', 'ephemeral-hdfs', 'persistent-hdfs', 'mapreduce', 'spark-standalone']
    if opts.hadoop_major_version == '1':
        modules = list(filter(lambda x: x != 'mapreduce', modules))
    if opts.ganglia:
        modules.append('ganglia')
    if opts.hadoop_major_version == 'yarn':
        opts.worker_instances = ''
    print('Cloning spark-ec2 scripts from {r}/tree/{b} on master...'.format(r=opts.spark_ec2_git_repo, b=opts.spark_ec2_git_branch))
    ssh(host=master, opts=opts, command='rm -rf spark-ec2' + ' && ' + 'git clone {r} -b {b} spark-ec2'.format(r=opts.spark_ec2_git_repo, b=opts.spark_ec2_git_branch))
    print('Deploying files to master...')
    deploy_files(conn=conn, root_dir=SPARK_EC2_DIR + '/' + 'deploy.generic', opts=opts, master_nodes=master_nodes, slave_nodes=slave_nodes, modules=modules)
    if opts.deploy_root_dir is not None:
        print('Deploying {s} to master...'.format(s=opts.deploy_root_dir))
        deploy_user_files(root_dir=opts.deploy_root_dir, opts=opts, master_nodes=master_nodes)
    print('Running setup on master...')
    setup_spark_cluster(master, opts)
    print('Done!')

def setup_spark_cluster(master, opts):
    if False:
        while True:
            i = 10
    ssh(master, opts, 'chmod u+x spark-ec2/setup.sh')
    ssh(master, opts, 'spark-ec2/setup.sh')
    print('Spark standalone cluster started at http://%s:8080' % master)
    if opts.ganglia:
        print('Ganglia started at http://%s:5080/ganglia' % master)

def is_ssh_available(host, opts, print_ssh_output=True):
    if False:
        print('Hello World!')
    '\n    Check if SSH is available on a host.\n    '
    s = subprocess.Popen(ssh_command(opts) + ['-t', '-t', '-o', 'ConnectTimeout=3', '%s@%s' % (opts.user, host), stringify_command('true')], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cmd_output = s.communicate()[0]
    if s.returncode != 0 and print_ssh_output:
        print(textwrap.dedent('\n\n            Warning: SSH connection error. (This could be temporary.)\n            Host: {h}\n            SSH return code: {r}\n            SSH output: {o}\n        ').format(h=host, r=s.returncode, o=cmd_output.strip()))
    return s.returncode == 0

def is_cluster_ssh_available(cluster_instances, opts):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if SSH is available on all the instances in a cluster.\n    '
    for i in cluster_instances:
        dns_name = get_dns_name(i, opts.private_ips)
        if not is_ssh_available(host=dns_name, opts=opts):
            return False
    else:
        return True

def wait_for_cluster_state(conn, opts, cluster_instances, cluster_state):
    if False:
        return 10
    "\n    Wait for all the instances in the cluster to reach a designated state.\n\n    cluster_instances: a list of boto.ec2.instance.Instance\n    cluster_state: a string representing the desired state of all the instances in the cluster\n           value can be 'ssh-ready' or a valid value from boto.ec2.instance.InstanceState such as\n           'running', 'terminated', etc.\n           (would be nice to replace this with a proper enum: http://stackoverflow.com/a/1695250)\n    "
    sys.stdout.write("Waiting for cluster to enter '{s}' state.".format(s=cluster_state))
    sys.stdout.flush()
    start_time = datetime.now()
    num_attempts = 0
    while True:
        time.sleep(5 * num_attempts)
        for i in cluster_instances:
            i.update()
        max_batch = 100
        statuses = []
        for j in xrange(0, len(cluster_instances), max_batch):
            batch = [i.id for i in cluster_instances[j:j + max_batch]]
            statuses.extend(conn.get_all_instance_status(instance_ids=batch))
        if cluster_state == 'ssh-ready':
            if all((i.state == 'running' for i in cluster_instances)) and all((s.system_status.status == 'ok' for s in statuses)) and all((s.instance_status.status == 'ok' for s in statuses)) and is_cluster_ssh_available(cluster_instances, opts):
                break
        elif all((i.state == cluster_state for i in cluster_instances)):
            break
        num_attempts += 1
        sys.stdout.write('.')
        sys.stdout.flush()
    sys.stdout.write('\n')
    end_time = datetime.now()
    print("Cluster is now in '{s}' state. Waited {t} seconds.".format(s=cluster_state, t=(end_time - start_time).seconds))

def get_num_disks(instance_type):
    if False:
        return 10
    disks_by_instance = {'c1.medium': 1, 'c1.xlarge': 4, 'c3.large': 2, 'c3.xlarge': 2, 'c3.2xlarge': 2, 'c3.4xlarge': 2, 'c3.8xlarge': 2, 'c4.large': 0, 'c4.xlarge': 0, 'c4.2xlarge': 0, 'c4.4xlarge': 0, 'c4.8xlarge': 0, 'cc1.4xlarge': 2, 'cc2.8xlarge': 4, 'cg1.4xlarge': 2, 'cr1.8xlarge': 2, 'd2.xlarge': 3, 'd2.2xlarge': 6, 'd2.4xlarge': 12, 'd2.8xlarge': 24, 'g2.2xlarge': 1, 'g2.8xlarge': 2, 'hi1.4xlarge': 2, 'hs1.8xlarge': 24, 'i2.xlarge': 1, 'i2.2xlarge': 2, 'i2.4xlarge': 4, 'i2.8xlarge': 8, 'm1.small': 1, 'm1.medium': 1, 'm1.large': 2, 'm1.xlarge': 4, 'm2.xlarge': 1, 'm2.2xlarge': 1, 'm2.4xlarge': 2, 'm3.medium': 1, 'm3.large': 1, 'm3.xlarge': 2, 'm3.2xlarge': 2, 'm4.large': 0, 'm4.xlarge': 0, 'm4.2xlarge': 0, 'm4.4xlarge': 0, 'm4.10xlarge': 0, 'p2.large': 0, 'p2.8xlarge': 0, 'p2.16xlarge': 0, 'r3.large': 1, 'r3.xlarge': 1, 'r3.2xlarge': 1, 'r3.4xlarge': 1, 'r3.8xlarge': 2, 't1.micro': 0, 't2.micro': 0, 't2.small': 0, 't2.medium': 0, 't2.large': 0}
    if instance_type in disks_by_instance:
        return disks_by_instance[instance_type]
    else:
        print("WARNING: Don't know number of disks on instance type %s; assuming 1" % instance_type, file=stderr)
        return 1

def deploy_files(conn, root_dir, opts, master_nodes, slave_nodes, modules):
    if False:
        return 10
    active_master = get_dns_name(master_nodes[0], opts.private_ips)
    num_disks = get_num_disks(opts.instance_type)
    hdfs_data_dirs = '/mnt/ephemeral-hdfs/data'
    mapred_local_dirs = '/mnt/hadoop/mrlocal'
    spark_local_dirs = '/mnt/spark'
    if num_disks > 1:
        for i in range(2, num_disks + 1):
            hdfs_data_dirs += ',/mnt%d/ephemeral-hdfs/data' % i
            mapred_local_dirs += ',/mnt%d/hadoop/mrlocal' % i
            spark_local_dirs += ',/mnt%d/spark' % i
    cluster_url = '%s:7077' % active_master
    if '.' in opts.spark_version:
        spark_v = get_validate_spark_version(opts.spark_version, opts.spark_git_repo)
        tachyon_v = get_tachyon_version(spark_v)
    else:
        spark_v = '%s|%s' % (opts.spark_git_repo, opts.spark_version)
        tachyon_v = ''
        print("Deploying Spark via git hash; Tachyon won't be set up")
        modules = filter(lambda x: x != 'tachyon', modules)
    master_addresses = [get_dns_name(i, opts.private_ips) for i in master_nodes]
    slave_addresses = [get_dns_name(i, opts.private_ips) for i in slave_nodes]
    worker_instances_str = '%d' % opts.worker_instances if opts.worker_instances else ''
    template_vars = {'master_list': '\n'.join(master_addresses), 'active_master': active_master, 'slave_list': '\n'.join(slave_addresses), 'cluster_url': cluster_url, 'hdfs_data_dirs': hdfs_data_dirs, 'mapred_local_dirs': mapred_local_dirs, 'spark_local_dirs': spark_local_dirs, 'swap': str(opts.swap), 'modules': '\n'.join(modules), 'spark_version': spark_v, 'tachyon_version': tachyon_v, 'hadoop_major_version': opts.hadoop_major_version, 'spark_worker_instances': worker_instances_str, 'spark_master_opts': opts.master_opts}
    if opts.copy_aws_credentials:
        template_vars['aws_access_key_id'] = conn.aws_access_key_id
        template_vars['aws_secret_access_key'] = conn.aws_secret_access_key
    else:
        template_vars['aws_access_key_id'] = ''
        template_vars['aws_secret_access_key'] = ''
    tmp_dir = tempfile.mkdtemp()
    for (path, dirs, files) in os.walk(root_dir):
        if path.find('.svn') == -1:
            dest_dir = os.path.join('/', path[len(root_dir):])
            local_dir = tmp_dir + dest_dir
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            for filename in files:
                if filename[0] not in '#.~' and filename[-1] != '~':
                    dest_file = os.path.join(dest_dir, filename)
                    local_file = tmp_dir + dest_file
                    with open(os.path.join(path, filename)) as src:
                        with open(local_file, 'w') as dest:
                            text = src.read()
                            for key in template_vars:
                                text = text.replace('{{' + key + '}}', template_vars[key])
                            dest.write(text)
                            dest.close()
    command = ['rsync', '-rv', '-e', stringify_command(ssh_command(opts)), '%s/' % tmp_dir, '%s@%s:/' % (opts.user, active_master)]
    subprocess.check_call(command)
    shutil.rmtree(tmp_dir)

def deploy_user_files(root_dir, opts, master_nodes):
    if False:
        return 10
    active_master = get_dns_name(master_nodes[0], opts.private_ips)
    command = ['rsync', '-rv', '-e', stringify_command(ssh_command(opts)), '%s' % root_dir, '%s@%s:/' % (opts.user, active_master)]
    subprocess.check_call(command)

def stringify_command(parts):
    if False:
        while True:
            i = 10
    if isinstance(parts, str):
        return parts
    else:
        return ' '.join(map(pipes.quote, parts))

def ssh_args(opts):
    if False:
        i = 10
        return i + 15
    parts = ['-o', 'StrictHostKeyChecking=no']
    parts += ['-o', 'UserKnownHostsFile=/dev/null']
    if opts.identity_file is not None:
        parts += ['-i', opts.identity_file]
    return parts

def ssh_command(opts):
    if False:
        i = 10
        return i + 15
    return ['ssh'] + ssh_args(opts)

def ssh(host, opts, command):
    if False:
        i = 10
        return i + 15
    tries = 0
    while True:
        try:
            return subprocess.check_call(ssh_command(opts) + ['-t', '-t', '%s@%s' % (opts.user, host), stringify_command(command)])
        except subprocess.CalledProcessError as e:
            if tries > 5:
                if e.returncode == 255:
                    raise UsageError('Failed to SSH to remote host {0}.\nPlease check that you have provided the correct --identity-file and --key-pair parameters and try again.'.format(host))
                else:
                    raise e
            print('Error executing remote command, retrying after 30 seconds: {0}'.format(e), file=stderr)
            time.sleep(30)
            tries = tries + 1

def _check_output(*popenargs, **kwargs):
    if False:
        return 10
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(*popenargs, stdout=subprocess.PIPE, **kwargs)
    (output, unused_err) = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get('args')
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd, output=output)
    return output

def ssh_read(host, opts, command):
    if False:
        i = 10
        return i + 15
    return _check_output(ssh_command(opts) + ['%s@%s' % (opts.user, host), stringify_command(command)])

def ssh_write(host, opts, command, arguments):
    if False:
        print('Hello World!')
    tries = 0
    while True:
        proc = subprocess.Popen(ssh_command(opts) + ['%s@%s' % (opts.user, host), stringify_command(command)], stdin=subprocess.PIPE)
        proc.stdin.write(arguments)
        proc.stdin.close()
        status = proc.wait()
        if status == 0:
            break
        elif tries > 5:
            raise RuntimeError('ssh_write failed with error %s' % proc.returncode)
        else:
            print('Error {0} while executing remote command, retrying after 30 seconds'.format(status), file=stderr)
            time.sleep(30)
            tries = tries + 1

def get_zones(conn, opts):
    if False:
        i = 10
        return i + 15
    if opts.zone == 'all':
        zones = [z.name for z in conn.get_all_zones()]
    else:
        zones = [opts.zone]
    return zones

def get_partition(total, num_partitions, current_partitions):
    if False:
        return 10
    num_slaves_this_zone = total // num_partitions
    if total % num_partitions - current_partitions > 0:
        num_slaves_this_zone += 1
    return num_slaves_this_zone

def get_ip_address(instance, private_ips=False):
    if False:
        while True:
            i = 10
    ip = instance.ip_address if not private_ips else instance.private_ip_address
    return ip

def get_dns_name(instance, private_ips=False):
    if False:
        for i in range(10):
            print('nop')
    dns = instance.public_dns_name if not private_ips else instance.private_ip_address
    return dns

def real_main():
    if False:
        while True:
            i = 10
    (opts, action, cluster_name) = parse_args()
    get_validate_spark_version(opts.spark_version, opts.spark_git_repo)
    if opts.wait is not None:
        warnings.warn('This option is deprecated and has no effect. spark-ec2 automatically waits as long as necessary for clusters to start up.', DeprecationWarning)
    if opts.identity_file is not None:
        if not os.path.exists(opts.identity_file):
            print("ERROR: The identity file '{f}' doesn't exist.".format(f=opts.identity_file), file=stderr)
            sys.exit(1)
        file_mode = os.stat(opts.identity_file).st_mode
        if not file_mode & S_IRUSR or not oct(file_mode)[-2:] == '00':
            print('ERROR: The identity file must be accessible only by you.', file=stderr)
            print('You can fix this with: chmod 400 "{f}"'.format(f=opts.identity_file), file=stderr)
            sys.exit(1)
    if opts.instance_type not in EC2_INSTANCE_TYPES:
        print('Warning: Unrecognized EC2 instance type for instance-type: {t}'.format(t=opts.instance_type), file=stderr)
    if opts.master_instance_type != '':
        if opts.master_instance_type not in EC2_INSTANCE_TYPES:
            print('Warning: Unrecognized EC2 instance type for master-instance-type: {t}'.format(t=opts.master_instance_type), file=stderr)
        if opts.instance_type in EC2_INSTANCE_TYPES and opts.master_instance_type in EC2_INSTANCE_TYPES:
            if EC2_INSTANCE_TYPES[opts.instance_type] != EC2_INSTANCE_TYPES[opts.master_instance_type]:
                print('Error: spark-ec2 currently does not support having a master and slaves with different AMI virtualization types.', file=stderr)
                print('master instance virtualization type: {t}'.format(t=EC2_INSTANCE_TYPES[opts.master_instance_type]), file=stderr)
                print('slave instance virtualization type: {t}'.format(t=EC2_INSTANCE_TYPES[opts.instance_type]), file=stderr)
                sys.exit(1)
    if opts.ebs_vol_num > 8:
        print('ebs-vol-num cannot be greater than 8', file=stderr)
        sys.exit(1)
    if opts.spark_ec2_git_repo.endswith('/') or opts.spark_ec2_git_repo.endswith('.git') or (not opts.spark_ec2_git_repo.startswith('https://github.com')) or (not opts.spark_ec2_git_repo.endswith('spark-ec2')):
        print('spark-ec2-git-repo must be a github repo and it must not have a trailing / or .git. Furthermore, we currently only support forks named spark-ec2.', file=stderr)
        sys.exit(1)
    if not (opts.deploy_root_dir is None or (os.path.isabs(opts.deploy_root_dir) and os.path.isdir(opts.deploy_root_dir) and os.path.exists(opts.deploy_root_dir))):
        print('--deploy-root-dir must be an absolute path to a directory that exists on the local file system', file=stderr)
        sys.exit(1)
    try:
        if opts.profile is None:
            conn = ec2.connect_to_region(opts.region)
        else:
            conn = ec2.connect_to_region(opts.region, profile_name=opts.profile)
    except Exception as e:
        print(e, file=stderr)
        sys.exit(1)
    if opts.zone == '':
        opts.zone = random.choice(conn.get_all_zones()).name
    if action == 'launch':
        if opts.slaves <= 0:
            print('ERROR: You have to start at least 1 slave', file=sys.stderr)
            sys.exit(1)
        if opts.resume:
            (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name)
        else:
            (master_nodes, slave_nodes) = launch_cluster(conn, opts, cluster_name)
        wait_for_cluster_state(conn=conn, opts=opts, cluster_instances=master_nodes + slave_nodes, cluster_state='ssh-ready')
        setup_cluster(conn, master_nodes, slave_nodes, opts, True)
    elif action == 'destroy':
        (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name, die_on_error=False)
        if any(master_nodes + slave_nodes):
            print('The following instances will be terminated:')
            for inst in master_nodes + slave_nodes:
                print('> %s' % get_dns_name(inst, opts.private_ips))
            print('ALL DATA ON ALL NODES WILL BE LOST!!')
        msg = 'Are you sure you want to destroy the cluster {c}? (y/N) '.format(c=cluster_name)
        response = raw_input(msg)
        if response == 'y':
            print('Terminating master...')
            for inst in master_nodes:
                inst.terminate()
            print('Terminating slaves...')
            for inst in slave_nodes:
                inst.terminate()
            if opts.delete_groups:
                group_names = [cluster_name + '-master', cluster_name + '-slaves']
                wait_for_cluster_state(conn=conn, opts=opts, cluster_instances=master_nodes + slave_nodes, cluster_state='terminated')
                print('Deleting security groups (this will take some time)...')
                attempt = 1
                while attempt <= 3:
                    print('Attempt %d' % attempt)
                    groups = [g for g in conn.get_all_security_groups() if g.name in group_names]
                    success = True
                    for group in groups:
                        print('Deleting rules in security group ' + group.name)
                        for rule in group.rules:
                            for grant in rule.grants:
                                success &= group.revoke(ip_protocol=rule.ip_protocol, from_port=rule.from_port, to_port=rule.to_port, src_group=grant)
                    time.sleep(30)
                    for group in groups:
                        try:
                            conn.delete_security_group(group_id=group.id)
                            print('Deleted security group %s' % group.name)
                        except boto.exception.EC2ResponseError:
                            success = False
                            print('Failed to delete security group %s' % group.name)
                    if success:
                        break
                    attempt += 1
                if not success:
                    print('Failed to delete all security groups after 3 tries.')
                    print('Try re-running in a few minutes.')
    elif action == 'login':
        (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name)
        if not master_nodes[0].public_dns_name and (not opts.private_ips):
            print('Master has no public DNS name.  Maybe you meant to specify --private-ips?')
        else:
            master = get_dns_name(master_nodes[0], opts.private_ips)
            print('Logging into master ' + master + '...')
            proxy_opt = []
            if opts.proxy_port is not None:
                proxy_opt = ['-D', opts.proxy_port]
            subprocess.check_call(ssh_command(opts) + proxy_opt + ['-t', '-t', '%s@%s' % (opts.user, master)])
    elif action == 'reboot-slaves':
        response = raw_input('Are you sure you want to reboot the cluster ' + cluster_name + ' slaves?\n' + 'Reboot cluster slaves ' + cluster_name + ' (y/N): ')
        if response == 'y':
            (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name, die_on_error=False)
            print('Rebooting slaves...')
            for inst in slave_nodes:
                if inst.state not in ['shutting-down', 'terminated']:
                    print('Rebooting ' + inst.id)
                    inst.reboot()
    elif action == 'get-master':
        (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name)
        if not master_nodes[0].public_dns_name and (not opts.private_ips):
            print('Master has no public DNS name.  Maybe you meant to specify --private-ips?')
        else:
            print(get_dns_name(master_nodes[0], opts.private_ips))
    elif action == 'stop':
        response = raw_input('Are you sure you want to stop the cluster ' + cluster_name + '?\nDATA ON EPHEMERAL DISKS WILL BE LOST, ' + 'BUT THE CLUSTER WILL KEEP USING SPACE ON\n' + 'AMAZON EBS IF IT IS EBS-BACKED!!\n' + 'All data on spot-instance slaves will be lost.\n' + 'Stop cluster ' + cluster_name + ' (y/N): ')
        if response == 'y':
            (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name, die_on_error=False)
            print('Stopping master...')
            for inst in master_nodes:
                if inst.state not in ['shutting-down', 'terminated']:
                    inst.stop()
            print('Stopping slaves...')
            for inst in slave_nodes:
                if inst.state not in ['shutting-down', 'terminated']:
                    if inst.spot_instance_request_id:
                        inst.terminate()
                    else:
                        inst.stop()
    elif action == 'start':
        (master_nodes, slave_nodes) = get_existing_cluster(conn, opts, cluster_name)
        print('Starting slaves...')
        for inst in slave_nodes:
            if inst.state not in ['shutting-down', 'terminated']:
                inst.start()
        print('Starting master...')
        for inst in master_nodes:
            if inst.state not in ['shutting-down', 'terminated']:
                inst.start()
        wait_for_cluster_state(conn=conn, opts=opts, cluster_instances=master_nodes + slave_nodes, cluster_state='ssh-ready')
        existing_master_type = master_nodes[0].instance_type
        existing_slave_type = slave_nodes[0].instance_type
        if existing_master_type == existing_slave_type:
            existing_master_type = ''
        opts.master_instance_type = existing_master_type
        opts.instance_type = existing_slave_type
        setup_cluster(conn, master_nodes, slave_nodes, opts, False)
    else:
        print('Invalid action: %s' % action, file=stderr)
        sys.exit(1)

def main():
    if False:
        for i in range(10):
            print('nop')
    try:
        real_main()
    except UsageError as e:
        print('\nError:\n', e, file=stderr)
        sys.exit(1)
if __name__ == '__main__':
    logging.basicConfig()
    main()