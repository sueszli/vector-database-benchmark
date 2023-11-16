from oso import polar_class

class Customer:
    pass

class InternalUser:
    ...

    def role(self):
        if False:
            i = 10
            return i + 15
        yield db.query('SELECT role FROM internal_roles WHERE id = ?', self.id)

class AccountManager(InternalUser):
    ...

    def customer_accounts(self):
        if False:
            i = 10
            return i + 15
        yield db.query('SELECT id FROM customer_accounts WHERE manager_id = ?', self.id)

def user_from_id(id):
    if False:
        return 10
    user_type = db.query('SELECT type FROM users WHERE id = ?', request.id)
    if user_type == 'internal':
        actor = InternalUser(request.id)
        if actor.role() == 'account_manager':
            return AccountManager(request.id)
        else:
            return actor
    elif user_type == 'customer':
        return Customer(request.id)