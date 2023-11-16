from security_monkey import app
from security_monkey.cloudaux_watcher import CloudAuxWatcher
from security_monkey.cloudaux_watcher import CloudAuxChangeItem
from security_monkey.decorators import record_exception
from cloudaux.decorators import iter_account_region

class CloudAuxBatchedWatcher(CloudAuxWatcher):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(CloudAuxBatchedWatcher, self).__init__(**kwargs)
        self.batched_size = 100
        self.done_slurping = False

    def slurp_list(self):
        if False:
            while True:
                i = 10
        self.prep_for_batch_slurp()

        @record_exception(source='{index}-watcher'.format(index=self.index), pop_exception_fields=True)
        def invoke_list_method(**kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return self.list_method(**kwargs['conn_dict'])

        @iter_account_region(self.service_name, accounts=self.account_identifiers, regions=self._get_regions(), conn_type='dict')
        def get_item_list(**kwargs):
            if False:
                i = 10
                return i + 15
            (kwargs, exception_map) = self._add_exception_fields_to_kwargs(**kwargs)
            items = invoke_list_method(**kwargs)
            if not items:
                items = list()
            return (items, exception_map)
        (items, exception_map) = self._flatten_iter_response(get_item_list())
        self.total_list.extend(items)
        if not items:
            self.done_slurping = True
        return (items, exception_map)

    def slurp(self):
        if False:
            for i in range(10):
                print('nop')

        @record_exception(source='{index}-watcher'.format(index=self.index), pop_exception_fields=True)
        def invoke_get_method(item, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return self.get_method(item, **kwargs['conn_dict'])

        @iter_account_region(self.service_name, accounts=self.account_identifiers, conn_type='dict', regions=['TBD'])
        def slurp_items(**kwargs):
            if False:
                for i in range(10):
                    print('nop')
            item_list = list()
            (kwargs, exception_map) = self._add_exception_fields_to_kwargs(**kwargs)
            item_counter = self.batch_counter * self.batched_size
            skip_counter = 0
            while self.batched_size - (len(item_list) + skip_counter) > 0 and (not self.done_slurping):
                cursor = self.total_list[item_counter]
                item_name = self.get_name_from_list_output(cursor)
                if item_name and self.check_ignore_list(item_name):
                    item_counter += 1
                    skip_counter += 1
                    if item_counter == len(self.total_list):
                        self.done_slurping = True
                    continue
                kwargs['conn_dict']['region'] = cursor['Region']
                app.logger.debug('Account: {account}, Batched Watcher: {watcher}, Fetching item: {item}/{region}'.format(account=kwargs['account_name'], watcher=self.index, item=item_name, region=kwargs['conn_dict']['region']))
                item_details = invoke_get_method(cursor, name=item_name, **kwargs)
                if item_details:
                    record_region = self.override_region or item_details.get('Region') or kwargs['conn_dict']['region']
                    item = CloudAuxChangeItem.from_item(name=item_name, item=item_details, record_region=record_region, source_watcher=self, **kwargs)
                    item_list.append(item)
                else:
                    skip_counter += 1
                item_counter += 1
                if item_counter == len(self.total_list):
                    self.done_slurping = True
            self.batch_counter += 1
            return (item_list, exception_map)
        return self._flatten_iter_response(slurp_items())