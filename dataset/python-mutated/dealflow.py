import singer
from mage_integrations.sources.pipedrive.tap_pipedrive.stream import PipedriveIterStream

class DealStageChangeStream(PipedriveIterStream):
    base_endpoint = 'deals'
    id_endpoint = 'deals/{}/flow'
    schema = 'dealflow'
    state_field = 'log_time'
    key_properties = ['id']
    replication_method = 'INCREMENTAL'

    def get_name(self):
        if False:
            for i in range(10):
                print('nop')
        return self.schema

    def process_row(self, row):
        if False:
            for i in range(10):
                print('nop')
        if row['object'] == 'dealChange':
            if row['data']['field_key'] == 'add_time' or row['data']['field_key'] == 'stage_id':
                return row['data']

    def update_endpoint(self, deal_id):
        if False:
            for i in range(10):
                print('nop')
        self.endpoint = self.id_endpoint.format(deal_id)