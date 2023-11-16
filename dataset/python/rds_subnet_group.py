#     Copyright 2016 Bridgewater Associates
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
"""
.. module: security_monkey.watchers.rds.rds_security_group
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.decorators import record_exception, iter_account_region
from security_monkey.watcher import Watcher
from security_monkey.watcher import ChangeItem
from security_monkey import app


class RDSSubnetGroup(Watcher):
    index = 'rdssubnetgroup'
    i_am_singular = 'RDS Subnet Group'
    i_am_plural = 'RDS Subnet Groups'

    def __init__(self, accounts=None, debug=False):
        super(RDSSubnetGroup, self).__init__(accounts=accounts, debug=debug)

    @record_exception()
    def describe_db_subnet_groups(self, **kwargs):
        from security_monkey.common.sts_connect import connect
        db_sub_groups = []
        rds = connect(kwargs['account_name'], 'boto3.rds.client', region=kwargs['region'],
                      assumed_role=kwargs['assumed_role'])

        marker = None

        while True:
            if marker:
                response = self.wrap_aws_rate_limited_call(
                    rds.describe_db_subnet_groups, Marker=marker)

            else:
                response = self.wrap_aws_rate_limited_call(
                    rds.describe_db_subnet_groups)

            db_sub_groups.extend(response.get('DBSubnetGroups'))
            if response.get('Marker'):
                marker = response.get('Marker')
            else:
                break

        return db_sub_groups

    def slurp(self):
        """
        :returns: item_list - list of RDS Subnet Groups.
        :returns: exception_map - A dict where the keys are a tuple containing the
            location of the exception and the value is the actual exception

        """
        self.prep_for_slurp()

        @iter_account_region(index=self.index, accounts=self.accounts, service_name='rds')
        def slurp_items(**kwargs):
            item_list = []
            exception_map = {}
            kwargs['exception_map'] = exception_map
            app.logger.debug("Checking {}/{}/{}".format(self.index,
                                                        kwargs['account_name'], kwargs['region']))
            db_sub_groups = self.describe_db_subnet_groups(**kwargs)

            if db_sub_groups:
                app.logger.debug("Found {} {}".format(
                    len(db_sub_groups), self.i_am_plural))
                for db_sub_group in db_sub_groups:

                    if self.check_ignore_list(db_sub_group.get('Name')):
                        continue

                    name = db_sub_group.get('DBSubnetGroupName')
                    vpc_id = None
                    if 'VpcId' in db_sub_group:
                        vpc_id = db_sub_group.get('VpcId')
                        name = "{} (in {})".format(name, vpc_id)

                    item_config = {
                        "name": name,
                        "db_subnet_group_description": db_sub_group.get('DBSubnetGroupDescription'),
                        "subnet_group_status": db_sub_group.get('SubnetGroupStatus'),
                        "vpc_id": db_sub_group.get('VpcId'),
                        "subnets": [],
                        "arn": db_sub_group.get('DBSubnetGroupArn')
                    }

                    for rds_subnet in db_sub_group.get('Subnets', []):
                        sub_config = {
                            "subnet_identifier": rds_subnet.get('SubnetIdentifier'),
                            "subnet_status": rds_subnet.get('SubnetStatus'),
                            "name": rds_subnet.get('SubnetAvailabilityZone', {}).get('Name'),
                        }
                        item_config["subnets"].append(sub_config)

                    item = RDSSubnetGroupItem(region=kwargs['region'],
                                              account=kwargs['account_name'],
                                              name=name, arn=item_config['arn'], config=item_config,
                                              source_watcher=self)

                    item_list.append(item)

            return item_list, exception_map
        return slurp_items()


class RDSSubnetGroupItem(ChangeItem):

    def __init__(self, region=None, account=None, name=None, arn=None, config=None, source_watcher=None):
        super(RDSSubnetGroupItem, self).__init__(
            index=RDSSubnetGroup.index,
            region=region,
            account=account,
            name=name,
            arn=arn,
            new_config=config if config else {},
            source_watcher=source_watcher)
