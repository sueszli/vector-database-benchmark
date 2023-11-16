from bigchaindb.common.exceptions import InvalidPowerChange
from bigchaindb.elections.election import Election
from bigchaindb.common.schema import TX_SCHEMA_VALIDATOR_ELECTION
from .validator_utils import new_validator_set, encode_validator, validate_asset_public_key

class ValidatorElection(Election):
    OPERATION = 'VALIDATOR_ELECTION'
    CREATE = OPERATION
    ALLOWED_OPERATIONS = (OPERATION,)
    TX_SCHEMA_CUSTOM = TX_SCHEMA_VALIDATOR_ELECTION

    def validate(self, bigchain, current_transactions=[]):
        if False:
            while True:
                i = 10
        'For more details refer BEP-21: https://github.com/bigchaindb/BEPs/tree/master/21\n        '
        current_validators = self.get_validators(bigchain)
        super(ValidatorElection, self).validate(bigchain, current_transactions=current_transactions)
        if self.asset['data']['power'] >= 1 / 3 * sum(current_validators.values()):
            raise InvalidPowerChange('`power` change must be less than 1/3 of total power')
        return self

    @classmethod
    def validate_schema(cls, tx):
        if False:
            i = 10
            return i + 15
        super(ValidatorElection, cls).validate_schema(tx)
        validate_asset_public_key(tx['asset']['data']['public_key'])

    def has_concluded(self, bigchain, *args, **kwargs):
        if False:
            while True:
                i = 10
        latest_block = bigchain.get_latest_block()
        if latest_block is not None:
            latest_block_height = latest_block['height']
            latest_validator_change = bigchain.get_validator_change()['height']
            if latest_validator_change == latest_block_height + 2:
                return False
        return super().has_concluded(bigchain, *args, **kwargs)

    def on_approval(self, bigchain, new_height):
        if False:
            i = 10
            return i + 15
        validator_updates = [self.asset['data']]
        curr_validator_set = bigchain.get_validators(new_height)
        updated_validator_set = new_validator_set(curr_validator_set, validator_updates)
        updated_validator_set = [v for v in updated_validator_set if v['voting_power'] > 0]
        bigchain.store_validator_set(new_height + 1, updated_validator_set)
        return encode_validator(self.asset['data'])

    def on_rollback(self, bigchaindb, new_height):
        if False:
            while True:
                i = 10
        bigchaindb.delete_validator_set(new_height + 1)