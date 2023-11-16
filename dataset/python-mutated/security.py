import odoo
import odoo.exceptions

def login(db, login, password):
    if False:
        for i in range(10):
            print('nop')
    res_users = odoo.registry(db)['res.users']
    return res_users._login(db, login, password)

def check(db, uid, passwd):
    if False:
        return 10
    res_users = odoo.registry(db)['res.users']
    return res_users.check(db, uid, passwd)