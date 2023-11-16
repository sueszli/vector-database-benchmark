import json
from bigchaindb.common.schema import TX_SCHEMA_CHAIN_MIGRATION_ELECTION
from bigchaindb.elections.election import Election

class ChainMigrationElection(Election):
    OPERATION = 'CHAIN_MIGRATION_ELECTION'
    CREATE = OPERATION
    ALLOWED_OPERATIONS = (OPERATION,)
    TX_SCHEMA_CUSTOM = TX_SCHEMA_CHAIN_MIGRATION_ELECTION

    def has_concluded(self, bigchaindb, *args, **kwargs):
        if False:
            return 10
        chain = bigchaindb.get_latest_abci_chain()
        if chain is not None and (not chain['is_synced']):
            return False
        return super().has_concluded(bigchaindb, *args, **kwargs)

    def on_approval(self, bigchain, *args, **kwargs):
        if False:
            print('Hello World!')
        bigchain.migrate_abci_chain()

    def show_election(self, bigchain):
        if False:
            return 10
        output = super().show_election(bigchain)
        chain = bigchain.get_latest_abci_chain()
        if chain is None or chain['is_synced']:
            return output
        output += f"\nchain_id={chain['chain_id']}"
        block = bigchain.get_latest_block()
        output += f"\napp_hash={block['app_hash']}"
        validators = [{'pub_key': {'type': 'tendermint/PubKeyEd25519', 'value': k}, 'power': v} for (k, v) in self.get_validators(bigchain).items()]
        output += f'\nvalidators={json.dumps(validators, indent=4)}'
        return output

    def on_rollback(self, bigchain, new_height):
        if False:
            print('Hello World!')
        bigchain.delete_abci_chain(new_height)