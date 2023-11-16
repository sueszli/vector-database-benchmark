import singer
from mage_integrations.sources.pipedrive.tap_pipedrive.stream import PipedriveIterStream

class DealsProductsStream(PipedriveIterStream):
    base_endpoint = 'deals'
    id_endpoint = 'deals/{}/products'
    schema = 'deal_products'
    state_field = None
    key_properties = ['id']

    def get_name(self):
        if False:
            return 10
        return self.schema

    def update_endpoint(self, deal_id):
        if False:
            i = 10
            return i + 15
        self.endpoint = self.id_endpoint.format(deal_id)