import re
from .. import If
from . import exactly_one, json_checker

def validate_json_checker(x):
    if False:
        i = 10
        return i + 15
    '\n    Property: BackupVault.AccessPolicy\n    '
    return json_checker(x)

def backup_vault_name(name):
    if False:
        while True:
            i = 10
    '\n    Property: BackupVault.BackupVaultName\n    '
    vault_name_re = re.compile('^[a-zA-Z0-9\\-\\_\\.]{1,50}$')
    if vault_name_re.match(name):
        return name
    else:
        raise ValueError('%s is not a valid backup vault name' % name)

def validate_backup_selection(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Class: BackupSelectionResourceType\n    '
    conds = ['ListOfTags', 'Resources']

    def check_if(names, props):
        if False:
            i = 10
            return i + 15
        validated = []
        for name in names:
            validated.append(name in props and isinstance(props[name], If))
        return all(validated)
    if check_if(conds, self.properties):
        return
    exactly_one(self.__class__.__name__, self.properties, conds)