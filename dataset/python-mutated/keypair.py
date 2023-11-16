"""
.. module: security_monkey.watchers.keypair
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Patrick Kelley <pkelley@netflix.com> @monkeysecurity

"""
from security_monkey.watcher import Watcher
from security_monkey.watcher import ChangeItem
from security_monkey.constants import TROUBLE_REGIONS
from security_monkey.exceptions import BotoConnectionIssue
from security_monkey.datastore import Account
from security_monkey import app, ARN_PREFIX

class Keypair(Watcher):
    index = 'keypair'
    i_am_singular = 'Keypair'
    i_am_plural = 'Keypairs'

    def __init__(self, accounts=None, debug=False):
        if False:
            return 10
        super(Keypair, self).__init__(accounts=accounts, debug=debug)

    def slurp(self):
        if False:
            while True:
                i = 10
        '\n        :returns: item_list - list of IAM SSH Keypairs.\n        :returns: exception_map - A dict where the keys are a tuple containing the\n            location of the exception and the value is the actual exception\n\n        '
        self.prep_for_slurp()
        item_list = []
        exception_map = {}
        from security_monkey.common.sts_connect import connect
        for account in self.accounts:
            try:
                account_db = Account.query.filter(Account.name == account).first()
                account_number = account_db.identifier
                ec2 = connect(account, 'ec2')
                regions = ec2.get_all_regions()
            except Exception as e:
                exc = BotoConnectionIssue(str(e), 'keypair', account, None)
                self.slurp_exception((self.index, account), exc, exception_map, source='{}-watcher'.format(self.index))
                continue
            for region in regions:
                app.logger.debug('Checking {}/{}/{}'.format(Keypair.index, account, region.name))
                try:
                    rec2 = connect(account, 'boto3.ec2.client', region=region)
                    kps = self.wrap_aws_rate_limited_call(rec2.describe_key_pairs)
                except Exception as e:
                    if region.name not in TROUBLE_REGIONS:
                        exc = BotoConnectionIssue(str(e), 'keypair', account, region.name)
                        self.slurp_exception((self.index, account, region.name), exc, exception_map, source='{}-watcher'.format(self.index))
                    continue
                app.logger.debug('Found {} {}'.format(len(kps), Keypair.i_am_plural))
                for kp in kps['KeyPairs']:
                    if self.check_ignore_list(kp['KeyName']):
                        continue
                    arn = ARN_PREFIX + ':ec2:{region}:{account_number}:key-pair/{name}'.format(region=region.name, account_number=account_number, name=kp['KeyName'])
                    item_list.append(KeypairItem(region=region.name, account=account, name=kp['KeyName'], arn=arn, config={'fingerprint': kp['KeyFingerprint'], 'arn': arn, 'name': kp['KeyName']}, source_watcher=self))
        return (item_list, exception_map)

class KeypairItem(ChangeItem):

    def __init__(self, region=None, account=None, name=None, arn=None, config=None, source_watcher=None):
        if False:
            for i in range(10):
                print('nop')
        super(KeypairItem, self).__init__(index=Keypair.index, region=region, account=account, name=name, arn=arn, new_config=config if config else {}, source_watcher=source_watcher)