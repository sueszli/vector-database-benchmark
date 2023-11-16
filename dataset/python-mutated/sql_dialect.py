from pyflink.java_gateway import get_gateway
__all__ = ['SqlDialect']

class SqlDialect(object):
    """
    Enumeration of valid SQL compatibility modes.

    In most of the cases, the built-in compatibility mode should be sufficient. For some features,
    i.e. the "INSERT INTO T PARTITION(a='xxx') ..." grammar, you may need to switch to the
    Hive dialect if required.

    We may introduce other SQL dialects in the future.

    :data:`DEFAULT`:

    Flink's default SQL behavior.

    :data:`HIVE`:

    SQL dialect that allows some Apache Hive specific grammar.

    Note: We might never support all of the Hive grammar. See the documentation for
    supported features.
    """
    DEFAULT = 0
    HIVE = 1

    @staticmethod
    def _from_j_sql_dialect(j_sql_dialect):
        if False:
            for i in range(10):
                print('nop')
        gateway = get_gateway()
        JSqlDialect = gateway.jvm.org.apache.flink.table.api.SqlDialect
        if j_sql_dialect == JSqlDialect.DEFAULT:
            return SqlDialect.DEFAULT
        elif j_sql_dialect == JSqlDialect.HIVE:
            return SqlDialect.HIVE
        else:
            raise Exception('Unsupported Java SQL dialect: %s' % j_sql_dialect)

    @staticmethod
    def _to_j_sql_dialect(sql_dialect):
        if False:
            while True:
                i = 10
        gateway = get_gateway()
        JSqlDialect = gateway.jvm.org.apache.flink.table.api.SqlDialect
        if sql_dialect == SqlDialect.DEFAULT:
            return JSqlDialect.DEFAULT
        elif sql_dialect == SqlDialect.HIVE:
            return JSqlDialect.HIVE
        else:
            raise TypeError('Unsupported SQL dialect: %s, supported SQL dialects are: SqlDialect.DEFAULT, SqlDialect.HIVE.' % sql_dialect)