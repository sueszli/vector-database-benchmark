from app import utils

def find_domain_by_scope_id(scope_id):
    if False:
        while True:
            i = 10
    query = {'scope_id': scope_id}
    items = utils.conn_db('asset_domain').distinct('domain', query)
    return list(items)