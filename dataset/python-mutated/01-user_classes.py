class Customer:

    def __init__(self, id):
        if False:
            for i in range(10):
                print('nop')
        self.id = id

class InternalUser:

    def __init__(self, id):
        if False:
            i = 10
            return i + 15
        self.id = id

def customer_dashboard_handler(request):
    if False:
        while True:
            i = 10
    oso = get_oso()
    actor = user_from_id(request.id)
    allowed = oso.is_allowed(actor=actor, action='view', resource='customer_dashboard')

def user_from_id(id):
    if False:
        return 10
    user_type = db.query('SELECT type FROM users WHERE id = ?', id)
    if user_type == 'internal':
        return InternalUser(id)
    elif user_type == 'customer':
        return Customer(id)