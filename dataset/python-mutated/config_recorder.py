"""
.. module: security_monkey.watchers.config_recorder
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.decorators import record_exception, iter_account_region
from security_monkey.watcher import Watcher
from security_monkey.watcher import ChangeItem
from security_monkey import app

class ConfigRecorder(Watcher):
    index = 'configrecorder'
    i_am_singular = 'Config Recorder'
    i_am_plural = 'Config Recorders'

    def __init__(self, accounts=None, debug=False):
        if False:
            for i in range(10):
                print('nop')
        super(ConfigRecorder, self).__init__(accounts=accounts, debug=debug)

    @record_exception(source='configrecorder')
    def describe_configuration_recorders(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        from security_monkey.common.sts_connect import connect
        config_service = connect(kwargs['account_name'], 'boto3.config.client', region=kwargs['region'])
        response = self.wrap_aws_rate_limited_call(config_service.describe_configuration_recorders)
        config_recorders = response.get('ConfigurationRecorders', [])
        return [recorder for recorder in config_recorders if not self.check_ignore_list(recorder.get('name'))]

    def slurp(self):
        if False:
            while True:
                i = 10
        '\n        :returns: item_list - list of AWS Config recorders.\n        :returns: exception_map - A dict where the keys are a tuple containing the\n            location of the exception and the value is the actual exception\n        '
        self.prep_for_slurp()

        @iter_account_region(index=self.index, accounts=self.accounts, service_name='config')
        def slurp_items(**kwargs):
            if False:
                for i in range(10):
                    print('nop')
            item_list = []
            exception_map = {}
            app.logger.debug('Checking {}/{}/{}'.format(self.index, kwargs['account_name'], kwargs['region']))
            config_recorders = self.describe_configuration_recorders(**kwargs)
            if config_recorders:
                app.logger.debug('Found {} {}.'.format(len(config_recorders), self.i_am_plural))
                for recorder in config_recorders:
                    name = recorder.get('name')
                    item_config = {'name': name, 'role_arn': recorder.get('roleARN'), 'recording_group': recorder.get('recordingGroup')}
                    item = ConfigRecorderItem(region=kwargs['region'], account=kwargs['account_name'], name=name, config=item_config, source_watcher=self)
                    item_list.append(item)
            return (item_list, exception_map)
        return slurp_items()

class ConfigRecorderItem(ChangeItem):

    def __init__(self, account=None, region=None, name=None, config=None, source_watcher=None):
        if False:
            print('Hello World!')
        super(ConfigRecorderItem, self).__init__(index=ConfigRecorder.index, region=region, account=account, name=name, new_config=config if config else {}, source_watcher=source_watcher)