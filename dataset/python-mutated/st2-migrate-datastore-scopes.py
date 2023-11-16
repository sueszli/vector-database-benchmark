from __future__ import absolute_import
import sys
import traceback as tb
from st2common import config
from st2common.constants.keyvalue import FULL_SYSTEM_SCOPE, SYSTEM_SCOPE
from st2common.constants.keyvalue import FULL_USER_SCOPE, USER_SCOPE
from st2common.models.db.keyvalue import KeyValuePairDB
from st2common.persistence.keyvalue import KeyValuePair
from st2common.service_setup import db_setup
from st2common.service_setup import db_teardown

def migrate_datastore():
    if False:
        i = 10
        return i + 15
    key_value_items = KeyValuePair.get_all()
    try:
        for kvp in key_value_items:
            kvp_id = getattr(kvp, 'id', None)
            secret = getattr(kvp, 'secret', False)
            scope = getattr(kvp, 'scope', SYSTEM_SCOPE)
            if scope == USER_SCOPE:
                scope = FULL_USER_SCOPE
            if scope == SYSTEM_SCOPE:
                scope = FULL_SYSTEM_SCOPE
            new_kvp_db = KeyValuePairDB(id=kvp_id, name=kvp.name, expire_timestamp=kvp.expire_timestamp, value=kvp.value, secret=secret, scope=scope)
            KeyValuePair.add_or_update(new_kvp_db)
    except:
        print('ERROR: Failed migrating datastore item with name: %s' % kvp.name)
        tb.print_exc()
        raise

def main():
    if False:
        return 10
    config.parse_args()
    db_setup()
    try:
        migrate_datastore()
        print('SUCCESS: Datastore items migrated successfully.')
        exit_code = 0
    except:
        print('ABORTED: Datastore migration aborted on first failure.')
        exit_code = 1
    db_teardown()
    sys.exit(exit_code)
if __name__ == '__main__':
    main()