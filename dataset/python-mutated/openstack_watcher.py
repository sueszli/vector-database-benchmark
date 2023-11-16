"""
.. module: security_monkey.openstack.watchers.openstack_watcher
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.cloudaux_watcher import CloudAuxWatcher, CloudAuxChangeItem
from security_monkey.decorators import record_exception
from cloudaux.openstack.decorators import iter_account_region, get_regions
from cloudaux.openstack.utils import list_items
from cloudaux.orchestration.openstack.utils import get_item

class OpenStackWatcher(CloudAuxWatcher):
    account_type = 'OpenStack'

    def __init__(self, accounts=None, debug=False):
        if False:
            while True:
                i = 10
        super(OpenStackWatcher, self).__init__(accounts=accounts, debug=debug)
        self.honor_ephemerals = True
        self.ephemeral_paths = ['updated_at']

    def _get_openstack_creds(self, account):
        if False:
            while True:
                i = 10
        from security_monkey.datastore import Account
        _account = Account.query.filter(Account.name.in_([account])).one()
        return (_account.identifier, _account.getCustom('cloudsyaml_file'))

    def _get_account_regions(self):
        if False:
            for i in range(10):
                print('nop')
        ' Regions are not global but account specific '

        def _get_regions(cloud_name, yaml_file):
            if False:
                return 10
            return [_region.get('name') for _region in get_regions(cloud_name, yaml_file)]
        account_regions = {}
        for account in self.accounts:
            (cloud_name, yaml_file) = self._get_openstack_creds(account)
            account_regions[account, cloud_name, yaml_file] = _get_regions(cloud_name, yaml_file)
        return account_regions

    def get_name_from_list_output(self, item):
        if False:
            i = 10
            return i + 15
        ' OpenStack allows for duplicate item names in same project for nearly all config types, add id '
        return '{} ({})'.format(item.name, item.id) if item.name else item.id

    def get_method(self, item, **kwargs):
        if False:
            return 10
        return get_item(item, **kwargs)

    def list_method(self, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['service'] = self.service
        kwargs['generator'] = self.generator
        return list_items(**kwargs)

    def _add_exception_fields_to_kwargs(self, **kwargs):
        if False:
            print('Hello World!')
        exception_map = dict()
        kwargs['index'] = self.index
        kwargs['account_name'] = kwargs['account_name']
        kwargs['exception_record_region'] = kwargs['region']
        kwargs['exception_map'] = exception_map
        return (kwargs, exception_map)

    def slurp(self):
        if False:
            for i in range(10):
                print('nop')
        self.prep_for_slurp()

        @record_exception(source='{index}-watcher'.format(index=self.index), pop_exception_fields=True)
        def invoke_list_method(**kwargs):
            if False:
                return 10
            return self.list_method(**kwargs)

        @record_exception(source='{index}-watcher'.format(index=self.index), pop_exception_fields=True)
        def invoke_get_method(item, **kwargs):
            if False:
                i = 10
                return i + 15
            return self.get_method(item, **kwargs)

        @iter_account_region(account_regions=self._get_account_regions())
        def slurp_items(**kwargs):
            if False:
                return 10
            (kwargs, exception_map) = self._add_exception_fields_to_kwargs(**kwargs)
            ' cache some of the kwargs in case they get popped before they are needed '
            region = kwargs['region']
            cloud_name = kwargs['cloud_name']
            account_name = kwargs['account_name']
            results = []
            item_list = invoke_list_method(**kwargs)
            if not item_list:
                return (results, exception_map)
            for item in item_list:
                item_name = self.get_name_from_list_output(item)
                if item_name and self.check_ignore_list(item_name):
                    continue
                item_details = invoke_get_method(item, **kwargs)
                if item_details:
                    arn = 'arn:openstack:{region}:{cloud_name}:{item_type}/{item_id}'.format(region=region, cloud_name=cloud_name, item_type=self.item_type, item_id=item.id)
                    item = OpenStackChangeItem(index=self.index, account=account_name, region=region, name=item_name, arn=arn, config=item_details)
                    results.append(item)
            return (results, exception_map)
        return self._flatten_iter_response(slurp_items())

class OpenStackChangeItem(CloudAuxChangeItem):

    def __init__(self, index=None, account=None, region=None, name=None, arn=None, config=None, source_watcher=None):
        if False:
            return 10
        super(OpenStackChangeItem, self).__init__(index=index, region=region, account=account, name=name, arn=arn, config=config, source_watcher=source_watcher)