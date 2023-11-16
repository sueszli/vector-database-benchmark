class CollectionResource:

    def on_post(self, req, resp, tenant_id, queue_name):
        if False:
            i = 10
            return i + 15
        pass

class ItemResource:

    def on_get(self, req, resp, tenant_id, queue_name, claim_id):
        if False:
            while True:
                i = 10
        pass

    def on_patch(self, req, resp, tenant_id, queue_name, claim_id):
        if False:
            return 10
        pass

    def on_delete(self, req, resp, tenant_id, queue_name, claim_id):
        if False:
            for i in range(10):
                print('nop')
        pass