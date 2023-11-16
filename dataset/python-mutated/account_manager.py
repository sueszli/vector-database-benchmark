"""
.. module: security_monkey.account
    :platform: Unix
    :synopsis: Base class for aws and other custom account types.


.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>

"""
from .datastore import Account, AccountType, AccountTypeCustomValues, User
from security_monkey import app, db
from security_monkey.common.utils import find_modules
import psycopg2
import traceback
from security_monkey.exceptions import AccountNameExists
account_registry = {}

class AccountManagerType(type):
    """
    Generates a global account registry as AccountManager derived classes
    are loaded
    """

    def __init__(cls, name, bases, attrs):
        if False:
            for i in range(10):
                print('nop')
        super(AccountManagerType, cls).__init__(name, bases, attrs)
        if cls.account_type:
            app.logger.info('Registering account %s %s', cls.account_type, cls.__name__)
            account_registry[cls.account_type] = cls

class CustomFieldConfig(object):
    """
    Defines additional field types for custom account types
    """

    def __init__(self, name, label, db_item, tool_tip, password=False, allowed_values=None):
        if False:
            for i in range(10):
                print('nop')
        super(CustomFieldConfig, self).__init__()
        self.name = name
        self.label = label
        self.db_item = db_item
        self.tool_tip = tool_tip
        self.password = password
        self.allowed_values = allowed_values

class AccountManager(object, metaclass=AccountManagerType):
    account_type = None
    compatable_account_types = []
    custom_field_configs = []
    identifier_label = None
    identifier_tool_tip = None

    def sanitize_account_identifier(self, identifier):
        if False:
            while True:
                i = 10
        'Each account type can determine how to sanitize the account identifier.\n        By default, will strip any whitespace.\n\n        Returns:\n            identifier stripped of whitespace\n        '
        return identifier.strip()

    def sanitize_account_name(self, name):
        if False:
            while True:
                i = 10
        'Each account type can determine how to sanitize the account name.\n        By default, will strip trailing whitespace.\n        Account alias (name) can have spaces and special characters\n\n        Returns:\n            name stripped of ending whitespace\n        '
        return name.rstrip()

    def sync(self, account_type, name, active, third_party, notes, identifier, custom_fields):
        if False:
            print('Hello World!')
        '\n        Syncs the account with the database. If account does not exist it is created. Other attributes\n        including account name are updated to conform with the third-party data source.\n        '
        account_type_result = _get_or_create_account_type(account_type)
        account = Account.query.filter(Account.identifier == identifier).first()
        if not account:
            account = Account()
        account = self._populate_account(account, account_type_result.id, self.sanitize_account_name(name), active, third_party, notes, self.sanitize_account_identifier(identifier), custom_fields)
        db.session.add(account)
        db.session.commit()
        db.session.refresh(account)
        account = self._load(account)
        db.session.expunge(account)
        return account

    def update(self, account_id, account_type, name, active, third_party, notes, identifier, custom_fields=None):
        if False:
            return 10
        '\n        Updates an existing account in the database.\n        '
        _get_or_create_account_type(account_type)
        if account_id:
            account = Account.query.filter(Account.id == account_id).first()
            if not account:
                app.logger.error('Account with ID {} does not exist.'.format(account_id))
                return None
            if account.name != name:
                if Account.query.filter(Account.name == name).first():
                    app.logger.error('Account with name: {} already exists.'.format(name))
                    raise AccountNameExists(name)
                account.name = self.sanitize_account_name(name)
        else:
            account = Account.query.filter(Account.name == name).first()
            if not account:
                app.logger.error('Account with name {} does not exist.'.format(name))
                return None
        account.active = active
        account.notes = notes
        account.active = active
        account.third_party = third_party
        account.identifier = self.sanitize_account_identifier(identifier)
        self._update_custom_fields(account, custom_fields)
        db.session.add(account)
        db.session.commit()
        db.session.refresh(account)
        account = self._load(account)
        db.session.expunge(account)
        return account

    def create(self, account_type, name, active, third_party, notes, identifier, custom_fields=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates an account in the database.\n        '
        account_type_result = _get_or_create_account_type(account_type)
        account = Account.query.filter(Account.name == name, Account.account_type_id == account_type_result.id).first()
        if account:
            app.logger.error('Account with name {} already exists!'.format(name))
            return None
        account = Account()
        account = self._populate_account(account, account_type_result.id, self.sanitize_account_name(name), active, third_party, notes, self.sanitize_account_identifier(identifier), custom_fields)
        db.session.add(account)
        db.session.commit()
        db.session.refresh(account)
        account = self._load(account)
        return account

    def lookup_account_by_identifier(self, identifier):
        if False:
            return 10
        query = Account.query.filter(Account.identifier == self.sanitize_account_identifier(identifier))
        if query.count():
            return query.first()
        else:
            return None

    def _load(self, account):
        if False:
            print('Hello World!')
        '\n        Placeholder for additional load related processing to be implemented\n        by account type specific subclasses\n        '
        return account

    def _populate_account(self, account, account_type_id, name, active, third_party, notes, identifier, custom_fields=None):
        if False:
            i = 10
            return i + 15
        '\n        Creates account DB object to be stored in the DB by create or update.\n        May be overridden to store additional data\n        '
        account.name = self.sanitize_account_name(name)
        account.identifier = self.sanitize_account_identifier(identifier)
        account.notes = notes
        account.active = active
        account.third_party = third_party
        account.account_type_id = account_type_id
        self._update_custom_fields(account, custom_fields)
        return account

    def _update_custom_fields(self, account, provided_custom_fields):
        if False:
            i = 10
            return i + 15
        existing_values = {}
        if account.custom_fields is None:
            account.custom_fields = []
        for cf in account.custom_fields:
            existing_values[cf.name] = cf
        for custom_config in self.custom_field_configs:
            if custom_config.db_item:
                new_value = None
                try:
                    if not provided_custom_fields.get(custom_config.name):
                        _ = existing_values[custom_config.name]
                    else:
                        new_value = provided_custom_fields[custom_config.name]
                        if existing_values[custom_config.name].value != new_value:
                            existing_values[custom_config.name].value = new_value
                            db.session.add(existing_values[custom_config.name])
                except KeyError:
                    new_custom_value = AccountTypeCustomValues(name=custom_config.name, value=new_value)
                    account.custom_fields.append(new_custom_value)
                    db.session.add(account)

    def is_compatible_with_account_type(self, account_type):
        if False:
            i = 10
            return i + 15
        if self.account_type == account_type or account_type in self.compatable_account_types:
            return True
        return False

def load_all_account_types():
    if False:
        for i in range(10):
            print('nop')
    ' Verifies all account types are in the database '
    for account_type in list(account_registry.keys()):
        _get_or_create_account_type(account_type)

def _get_or_create_account_type(account_type):
    if False:
        for i in range(10):
            print('nop')
    account_type_result = AccountType.query.filter(AccountType.name == account_type).first()
    if not account_type_result:
        account_type_result = AccountType(name=account_type)
        db.session.add(account_type_result)
        db.session.commit()
        app.logger.info('Creating a new AccountType: {} - ID: {}'.format(account_type, account_type_result.id))
    return account_type_result

def get_account_by_id(account_id):
    if False:
        for i in range(10):
            print('nop')
    '\n    Retrieves an account plus any additional custom fields\n    '
    account = Account.query.filter(Account.id == account_id).first()
    manager_class = account_registry.get(account.account_type.name)
    account = manager_class()._load(account)
    db.session.expunge(account)
    return account

def get_account_by_name(account_name):
    if False:
        while True:
            i = 10
    '\n    Retrieves an account plus any additional custom fields\n    '
    account = Account.query.filter(Account.name == account_name).first()
    manager_class = account_registry.get(account.account_type.name)
    account = manager_class()._load(account)
    db.session.expunge(account)
    return account

def delete_account_by_id(account_id):
    if False:
        while True:
            i = 10
    users = User.query.filter(User.accounts.any(Account.id == account_id)).all()
    for user in users:
        user.accounts = [account for account in user.accounts if not account.id == account_id]
        db.session.add(user)
        db.session.commit()
    conn = None
    try:
        conn = psycopg2.connect(app.config.get('SQLALCHEMY_DATABASE_URI'))
        cur = conn.cursor()
        cur.execute('DELETE from issue_item_association WHERE super_issue_id IN (SELECT itemaudit.id from itemaudit, item WHERE itemaudit.item_id = item.id AND item.account_id = %s);', [account_id])
        cur.execute('DELETE from itemaudit WHERE item_id IN (SELECT id from item WHERE account_id = %s);', [account_id])
        cur.execute('DELETE from itemrevisioncomment WHERE revision_id IN (SELECT itemrevision.id from itemrevision, item WHERE itemrevision.item_id = item.id AND item.account_id = %s);', [account_id])
        cur.execute('DELETE from cloudtrail WHERE revision_id IN (SELECT itemrevision.id from itemrevision, item WHERE itemrevision.item_id = item.id AND item.account_id = %s);', [account_id])
        cur.execute('DELETE from itemrevision WHERE item_id IN (SELECT id from item WHERE account_id = %s);', [account_id])
        cur.execute('DELETE from itemcomment WHERE item_id IN (SELECT id from item WHERE account_id = %s);', [account_id])
        cur.execute('DELETE from exceptions WHERE item_id IN (SELECT id from item WHERE account_id = %s);', [account_id])
        cur.execute('DELETE from cloudtrail WHERE item_id IN (SELECT id from item WHERE account_id = %s);', [account_id])
        cur.execute('DELETE from item WHERE account_id = %s;', [account_id])
        cur.execute('DELETE from exceptions WHERE account_id = %s;', [account_id])
        cur.execute('DELETE from auditorsettings WHERE account_id = %s;', [account_id])
        cur.execute('DELETE from account_type_values WHERE account_id = %s;', [account_id])
        cur.execute('DELETE from account WHERE id = %s;', [account_id])
        conn.commit()
    except Exception as e:
        app.logger.warn(traceback.format_exc())
    finally:
        if conn:
            conn.close()

def delete_account_by_name(name):
    if False:
        while True:
            i = 10
    account = Account.query.filter(Account.name == name).first()
    account_id = account.id
    db.session.expunge(account)
    delete_account_by_id(account_id)

def bulk_disable_accounts(account_names):
    if False:
        while True:
            i = 10
    'Bulk disable accounts'
    for account_name in account_names:
        account = Account.query.filter(Account.name == account_name).first()
        if account:
            app.logger.debug('Disabling account %s', account.name)
            account.active = False
            db.session.add(account)
    db.session.commit()
    db.session.close()

def bulk_enable_accounts(account_names):
    if False:
        while True:
            i = 10
    'Bulk enable accounts'
    for account_name in account_names:
        account = Account.query.filter(Account.name == account_name).first()
        if account:
            app.logger.debug('Enabling account %s', account.name)
            account.active = True
            db.session.add(account)
    db.session.commit()
    db.session.close()
find_modules('account_managers')