"""
.. module: security_monkey.views.account_bulk_update
    :platform: Unix
    :synopsis: Updates the active flag for a list of accounts.


.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.views import AuthenticatedService
from security_monkey.datastore import Account
from security_monkey import app, db, rbac
from flask import request
from flask_restful import reqparse
import json

class AccountListPut(AuthenticatedService):
    decorators = [rbac.allow(['Admin'], ['PUT'])]

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(AccountListPut, self).__init__()
        self.reqparse = reqparse.RequestParser()

    def put(self):
        if False:
            i = 10
            return i + 15
        values = json.loads(request.json)
        app.logger.debug('Account bulk update {}'.format(values))
        for account_name in list(values.keys()):
            account = Account.query.filter(Account.name == account_name).first()
            if account:
                account.active = values[account_name]
                db.session.add(account)
        db.session.commit()
        db.session.close()
        return ({'status': 'updated'}, 200)