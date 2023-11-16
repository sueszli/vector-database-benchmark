from bigchaindb.utils import condition_details_has_owner
from bigchaindb.backend import query
from bigchaindb.common.transaction import TransactionLink

class FastQuery:
    """Database queries that join on block results from a single node."""

    def __init__(self, connection):
        if False:
            while True:
                i = 10
        self.connection = connection

    def get_outputs_by_public_key(self, public_key):
        if False:
            return 10
        'Get outputs for a public key'
        txs = list(query.get_owned_ids(self.connection, public_key))
        return [TransactionLink(tx['id'], index) for tx in txs for (index, output) in enumerate(tx['outputs']) if condition_details_has_owner(output['condition']['details'], public_key)]

    def filter_spent_outputs(self, outputs):
        if False:
            i = 10
            return i + 15
        'Remove outputs that have been spent\n\n        Args:\n            outputs: list of TransactionLink\n        '
        links = [o.to_dict() for o in outputs]
        txs = list(query.get_spending_transactions(self.connection, links))
        spends = {TransactionLink.from_dict(input_['fulfills']) for tx in txs for input_ in tx['inputs']}
        return [ff for ff in outputs if ff not in spends]

    def filter_unspent_outputs(self, outputs):
        if False:
            while True:
                i = 10
        'Remove outputs that have not been spent\n\n        Args:\n            outputs: list of TransactionLink\n        '
        links = [o.to_dict() for o in outputs]
        txs = list(query.get_spending_transactions(self.connection, links))
        spends = {TransactionLink.from_dict(input_['fulfills']) for tx in txs for input_ in tx['inputs']}
        return [ff for ff in outputs if ff in spends]