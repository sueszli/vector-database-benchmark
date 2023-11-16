import ipaddress
from functools import lru_cache
try:
    from psycopg import ClientCursor, IsolationLevel, adapt, adapters, errors, sql
    from psycopg.postgres import types
    from psycopg.types.datetime import TimestamptzLoader
    from psycopg.types.json import Jsonb
    from psycopg.types.range import Range, RangeDumper
    from psycopg.types.string import TextLoader
    Inet = ipaddress.ip_address
    DateRange = DateTimeRange = DateTimeTZRange = NumericRange = Range
    RANGE_TYPES = (Range,)
    TSRANGE_OID = types['tsrange'].oid
    TSTZRANGE_OID = types['tstzrange'].oid

    def mogrify(sql, params, connection):
        if False:
            i = 10
            return i + 15
        with connection.cursor() as cursor:
            return ClientCursor(cursor.connection).mogrify(sql, params)

    class BaseTzLoader(TimestamptzLoader):
        """
        Load a PostgreSQL timestamptz using the a specific timezone.
        The timezone can be None too, in which case it will be chopped.
        """
        timezone = None

        def load(self, data):
            if False:
                while True:
                    i = 10
            res = super().load(data)
            return res.replace(tzinfo=self.timezone)

    def register_tzloader(tz, context):
        if False:
            return 10

        class SpecificTzLoader(BaseTzLoader):
            timezone = tz
        context.adapters.register_loader('timestamptz', SpecificTzLoader)

    class DjangoRangeDumper(RangeDumper):
        """A Range dumper customized for Django."""

        def upgrade(self, obj, format):
            if False:
                while True:
                    i = 10
            dumper = super().upgrade(obj, format)
            if dumper is not self and dumper.oid == TSRANGE_OID:
                dumper.oid = TSTZRANGE_OID
            return dumper

    @lru_cache
    def get_adapters_template(use_tz, timezone):
        if False:
            for i in range(10):
                print('nop')
        ctx = adapt.AdaptersMap(adapters)
        ctx.register_loader('jsonb', TextLoader)
        ctx.register_loader('inet', TextLoader)
        ctx.register_loader('cidr', TextLoader)
        ctx.register_dumper(Range, DjangoRangeDumper)
        register_tzloader(timezone, ctx)
        return ctx
    is_psycopg3 = True
except ImportError:
    from enum import IntEnum
    from psycopg2 import errors, extensions, sql
    from psycopg2.extras import DateRange, DateTimeRange, DateTimeTZRange, Inet
    from psycopg2.extras import Json as Jsonb
    from psycopg2.extras import NumericRange, Range
    RANGE_TYPES = (DateRange, DateTimeRange, DateTimeTZRange, NumericRange)

    class IsolationLevel(IntEnum):
        READ_UNCOMMITTED = extensions.ISOLATION_LEVEL_READ_UNCOMMITTED
        READ_COMMITTED = extensions.ISOLATION_LEVEL_READ_COMMITTED
        REPEATABLE_READ = extensions.ISOLATION_LEVEL_REPEATABLE_READ
        SERIALIZABLE = extensions.ISOLATION_LEVEL_SERIALIZABLE

    def _quote(value, connection=None):
        if False:
            while True:
                i = 10
        adapted = extensions.adapt(value)
        if hasattr(adapted, 'encoding'):
            adapted.encoding = 'utf8'
        return adapted.getquoted().decode()
    sql.quote = _quote

    def mogrify(sql, params, connection):
        if False:
            while True:
                i = 10
        with connection.cursor() as cursor:
            return cursor.mogrify(sql, params).decode()
    is_psycopg3 = False