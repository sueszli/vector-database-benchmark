from frappe import _

def get_data():
    if False:
        for i in range(10):
            print('nop')
    return {'fieldname': 'user_type', 'transactions': [{'label': _('Reference'), 'items': ['User']}]}