"""
.. module: security_monkey.watchers.vpc.nat_gateway
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.watcher import Watcher
from security_monkey.watcher import ChangeItem
from security_monkey.constants import TROUBLE_REGIONS
from security_monkey.exceptions import BotoConnectionIssue
from security_monkey import app
from boto.vpc import regions

class NATGateway(Watcher):
    index = 'natgateway'
    i_am_singular = 'NAT Gateway'
    i_am_plural = 'NAT Gateways'

    def __init__(self, accounts=None, debug=False):
        if False:
            print('Hello World!')
        super(NATGateway, self).__init__(accounts=accounts, debug=debug)

    def slurp(self):
        if False:
            i = 10
            return i + 15
        '\n        :returns: item_list - list of nat_gateways.\n        :returns: exception_map - A dict where the keys are a tuple containing the\n            location of the exception and the value is the actual exception\n\n        '
        self.prep_for_slurp()
        item_list = []
        exception_map = {}
        from security_monkey.common.sts_connect import connect
        for account in self.accounts:
            for region in regions():
                app.logger.debug('Checking {}/{}/{}'.format(self.index, account, region.name))
                try:
                    conn = connect(account, 'boto3.ec2.client', region=region)
                    all_nat_gateways_resp = self.wrap_aws_rate_limited_call(conn.describe_nat_gateways)
                    all_nat_gateways = all_nat_gateways_resp.get('NatGateways', [])
                except Exception as e:
                    if region.name not in TROUBLE_REGIONS:
                        exc = BotoConnectionIssue(str(e), self.index, account, region.name)
                        self.slurp_exception((self.index, account, region.name), exc, exception_map)
                    continue
                app.logger.debug('Found {} {}'.format(len(all_nat_gateways), self.i_am_plural))
                for nat_gateway in all_nat_gateways:
                    nat_gateway_name = nat_gateway.get('NatGatewayId')
                    if self.check_ignore_list(nat_gateway_name):
                        continue
                    natGatewayAddresses = []
                    for address in nat_gateway.get('NatGatewayAddresses'):
                        next_addr = {'public_ip': address.get('PublicIp', None), 'allocation_id': address.get('AllocationId', None), 'private_ip': address.get('PrivateIp', None), 'network_interface_id': address.get('NetworkInterfaceId', None)}
                        natGatewayAddresses.append(next_addr)
                    config = {'id': nat_gateway.get('NatGatewayId'), 'subnet_id': nat_gateway.get('SubnetId'), 'vpc_id': nat_gateway.get('VpcId'), 'create_time': str(nat_gateway.get('CreateTime')), 'delete_time': str(nat_gateway.get('DeleteTime')), 'nat_gateway_addresses': natGatewayAddresses, 'state': nat_gateway.get('State'), 'failure_code': nat_gateway.get('FailureCode'), 'failure_message': nat_gateway.get('FailureMessage')}
                    item = NATGatewayItem(region=region.name, account=account, name=nat_gateway_name, config=config, source_watcher=self)
                    item_list.append(item)
        return (item_list, exception_map)

class NATGatewayItem(ChangeItem):

    def __init__(self, region=None, account=None, name=None, config=None, source_watcher=None):
        if False:
            i = 10
            return i + 15
        super(NATGatewayItem, self).__init__(index=NATGateway.index, region=region, account=account, name=name, new_config=config if config else {}, source_watcher=source_watcher)