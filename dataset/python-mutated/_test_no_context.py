import frappe

def get_context():
    if False:
        while True:
            i = 10
    context = frappe._dict()
    context.body = 'Custom Content'
    return context