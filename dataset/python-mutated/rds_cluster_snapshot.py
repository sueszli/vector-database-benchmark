"""
.. module: security_monkey.watchers.rds.rds_cluster_snapshot
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.watcher import Watcher
from security_monkey.watcher import ChangeItem
from security_monkey.constants import TROUBLE_REGIONS
from security_monkey.exceptions import BotoConnectionIssue
from security_monkey import app
from boto.rds import regions

class RDSClusterSnapshot(Watcher):
    index = 'rdsclustersnapshot'
    i_am_singular = 'RDS Cluster Snapshot'
    i_am_plural = 'RDS Cluster Snapshots'

    def __init__(self, accounts=None, debug=False):
        if False:
            while True:
                i = 10
        super(RDSClusterSnapshot, self).__init__(accounts=accounts, debug=debug)

    def slurp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :returns: item_list - list of RDS Cluster Snapshots.\n        :returns: exception_map - A dict where the keys are a tuple containing the\n            location of the exception and the value is the actual exception\n\n        '
        self.prep_for_slurp()
        from security_monkey.common.sts_connect import connect
        item_list = []
        exception_map = {}
        for account in self.accounts:
            for region in regions():
                app.logger.debug('Checking {}/{}/{}'.format(self.index, account, region.name))
                rds_cluster_snapshots = []
                try:
                    rds = connect(account, 'boto3.rds.client', region=region)
                    marker = None
                    while True:
                        if marker:
                            response = self.wrap_aws_rate_limited_call(rds.describe_db_cluster_snapshots, Marker=marker)
                        else:
                            response = self.wrap_aws_rate_limited_call(rds.describe_db_cluster_snapshots)
                        rds_cluster_snapshots.extend(response.get('DBClusterSnapshots'))
                        if response.get('Marker'):
                            marker = response.get('Marker')
                        else:
                            break
                except Exception as e:
                    if region.name not in TROUBLE_REGIONS:
                        exc = BotoConnectionIssue(str(e), self.index, account, region.name)
                        self.slurp_exception((self.index, account, region.name), exc, exception_map)
                    continue
                app.logger.debug('Found {} {}'.format(len(rds_cluster_snapshots), self.i_am_plural))
                for cluster_snapshot in rds_cluster_snapshots:
                    name = cluster_snapshot.get('DBClusterSnapshotIdentifier')
                    if self.check_ignore_list(name):
                        continue
                    config = {'db_cluster_snapshot_identifier': name, 'db_cluster_identifier': cluster_snapshot.get('DBClusterIdentifier'), 'snapshot_create_time': str(cluster_snapshot.get('SnapshotCreateTime')), 'availability_zones': cluster_snapshot.get('AvailabilityZones'), 'engine': cluster_snapshot.get('Engine'), 'allocated_storage': cluster_snapshot.get('AllocatedStorage'), 'status': cluster_snapshot.get('Status'), 'port': cluster_snapshot.get('Port'), 'vpc_id': cluster_snapshot.get('VpcId'), 'cluster_create_time': str(cluster_snapshot.get('ClusterCreateTime')), 'master_username': cluster_snapshot.get('MasterUsername'), 'engine_version': cluster_snapshot.get('EngineVersion'), 'license_model': cluster_snapshot.get('LicenseModel'), 'snapshot_type': cluster_snapshot.get('SnapshotType'), 'percent_progress': cluster_snapshot.get('PercentProgress'), 'storage_encrypted': cluster_snapshot.get('StorageEncrypted'), 'kms_key_id': cluster_snapshot.get('KmsKeyId')}
                    item = RDSClusterSnapshotItem(region=region.name, account=account, name=name, config=dict(config), source_watcher=self)
                    item_list.append(item)
        return (item_list, exception_map)

class RDSClusterSnapshotItem(ChangeItem):

    def __init__(self, region=None, account=None, name=None, config=None, source_watcher=None):
        if False:
            i = 10
            return i + 15
        super(RDSClusterSnapshotItem, self).__init__(index=RDSClusterSnapshot.index, region=region, account=account, name=name, new_config=config if config else {}, source_watcher=source_watcher)