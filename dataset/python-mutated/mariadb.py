from .base import MariaDBIdentifierPreparer
from .base import MySQLDialect

class MariaDBDialect(MySQLDialect):
    is_mariadb = True
    supports_statement_cache = True
    name = 'mariadb'
    preparer = MariaDBIdentifierPreparer

def loader(driver):
    if False:
        for i in range(10):
            print('nop')
    driver_mod = __import__('sqlalchemy.dialects.mysql.%s' % driver).dialects.mysql
    driver_cls = getattr(driver_mod, driver).dialect
    return type('MariaDBDialect_%s' % driver, (MariaDBDialect, driver_cls), {'supports_statement_cache': True})