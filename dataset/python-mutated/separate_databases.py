"""Illustrates sharding using distinct SQLite databases."""
from __future__ import annotations
import datetime
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import ForeignKey
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import select
from sqlalchemy import Table
from sqlalchemy.ext.horizontal_shard import set_shard_id
from sqlalchemy.ext.horizontal_shard import ShardedSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import operators
from sqlalchemy.sql import visitors
echo = True
db1 = create_engine('sqlite://', echo=echo)
db2 = create_engine('sqlite://', echo=echo)
db3 = create_engine('sqlite://', echo=echo)
db4 = create_engine('sqlite://', echo=echo)
Session = sessionmaker(class_=ShardedSession, shards={'north_america': db1, 'asia': db2, 'europe': db3, 'south_america': db4})

class Base(DeclarativeBase):
    pass
ids = Table('ids', Base.metadata, Column('nextid', Integer, nullable=False))

def id_generator(ctx):
    if False:
        for i in range(10):
            print('nop')
    with db1.begin() as conn:
        nextid = conn.scalar(ids.select().with_for_update())
        conn.execute(ids.update().values({ids.c.nextid: ids.c.nextid + 1}))
    return nextid

class WeatherLocation(Base):
    __tablename__ = 'weather_locations'
    id: Mapped[int] = mapped_column(primary_key=True, default=id_generator)
    continent: Mapped[str]
    city: Mapped[str]
    reports: Mapped[list[Report]] = relationship(back_populates='location')

    def __init__(self, continent: str, city: str):
        if False:
            i = 10
            return i + 15
        self.continent = continent
        self.city = city

class Report(Base):
    __tablename__ = 'weather_reports'
    id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int] = mapped_column(ForeignKey('weather_locations.id'))
    temperature: Mapped[float]
    report_time: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)
    location: Mapped[WeatherLocation] = relationship(back_populates='reports')

    def __init__(self, temperature: float):
        if False:
            while True:
                i = 10
        self.temperature = temperature
shard_lookup = {'North America': 'north_america', 'Asia': 'asia', 'Europe': 'europe', 'South America': 'south_america'}

def shard_chooser(mapper, instance, clause=None):
    if False:
        while True:
            i = 10
    "shard chooser.\n\n    looks at the given instance and returns a shard id\n    note that we need to define conditions for\n    the WeatherLocation class, as well as our secondary Report class which will\n    point back to its WeatherLocation via its 'location' attribute.\n\n    "
    if isinstance(instance, WeatherLocation):
        return shard_lookup[instance.continent]
    else:
        return shard_chooser(mapper, instance.location)

def identity_chooser(mapper, primary_key, *, lazy_loaded_from, **kw):
    if False:
        for i in range(10):
            print('nop')
    "identity chooser.\n\n    given a primary key, returns a list of shards\n    to search.  here, we don't have any particular information from a\n    pk so we just return all shard ids. often, you'd want to do some\n    kind of round-robin strategy here so that requests are evenly\n    distributed among DBs.\n\n    "
    if lazy_loaded_from:
        return [lazy_loaded_from.identity_token]
    else:
        return ['north_america', 'asia', 'europe', 'south_america']

def execute_chooser(context):
    if False:
        for i in range(10):
            print('nop')
    "statement execution chooser.\n\n    this also returns a list of shard ids, which can just be all of them. but\n    here we'll search into the execution context in order to try to narrow down\n    the list of shards to SELECT.\n\n    "
    ids = []
    for (column, operator, value) in _get_select_comparisons(context.statement):
        if column.shares_lineage(WeatherLocation.__table__.c.continent):
            if operator == operators.eq:
                ids.append(shard_lookup[value])
            elif operator == operators.in_op:
                ids.extend((shard_lookup[v] for v in value))
    if len(ids) == 0:
        return ['north_america', 'asia', 'europe', 'south_america']
    else:
        return ids

def _get_select_comparisons(statement):
    if False:
        print('Hello World!')
    'Search a Select or Query object for binary expressions.\n\n    Returns expressions which match a Column against one or more\n    literal values as a list of tuples of the form\n    (column, operator, values).   "values" is a single value\n    or tuple of values depending on the operator.\n\n    '
    binds = {}
    clauses = set()
    comparisons = []

    def visit_bindparam(bind):
        if False:
            print('Hello World!')
        value = bind.effective_value
        binds[bind] = value

    def visit_column(column):
        if False:
            return 10
        clauses.add(column)

    def visit_binary(binary):
        if False:
            for i in range(10):
                print('nop')
        if binary.left in clauses and binary.right in binds:
            comparisons.append((binary.left, binary.operator, binds[binary.right]))
        elif binary.left in binds and binary.right in clauses:
            comparisons.append((binary.right, binary.operator, binds[binary.left]))
    if statement.whereclause is not None:
        visitors.traverse(statement.whereclause, {}, {'bindparam': visit_bindparam, 'binary': visit_binary, 'column': visit_column})
    return comparisons
Session.configure(shard_chooser=shard_chooser, identity_chooser=identity_chooser, execute_chooser=execute_chooser)

def setup():
    if False:
        return 10
    for db in (db1, db2, db3, db4):
        Base.metadata.create_all(db)
    with db1.begin() as conn:
        conn.execute(ids.insert(), {'nextid': 1})

def main():
    if False:
        return 10
    setup()
    tokyo = WeatherLocation('Asia', 'Tokyo')
    newyork = WeatherLocation('North America', 'New York')
    toronto = WeatherLocation('North America', 'Toronto')
    london = WeatherLocation('Europe', 'London')
    dublin = WeatherLocation('Europe', 'Dublin')
    brasilia = WeatherLocation('South America', 'Brasila')
    quito = WeatherLocation('South America', 'Quito')
    tokyo.reports.append(Report(80.0))
    newyork.reports.append(Report(75))
    quito.reports.append(Report(85))
    with Session() as sess:
        sess.add_all([tokyo, newyork, toronto, london, dublin, brasilia, quito])
        sess.commit()
        t = sess.get(WeatherLocation, tokyo.id)
        assert t.city == tokyo.city
        assert t.reports[0].temperature == 80.0
        asia_and_europe = sess.execute(select(WeatherLocation).filter(WeatherLocation.continent.in_(['Europe', 'Asia']))).scalars()
        assert {c.city for c in asia_and_europe} == {'Tokyo', 'London', 'Dublin'}
        north_american_cities_w_t = sess.execute(select(WeatherLocation).filter(WeatherLocation.city.startswith('T')).options(set_shard_id('north_america'))).scalars()
        assert {c.city for c in north_american_cities_w_t} == {'Toronto'}
        newyork_report = newyork.reports[0]
        tokyo_report = tokyo.reports[0]
        assert inspect(newyork_report).identity_key == (Report, (1,), 'north_america')
        assert inspect(tokyo_report).identity_key == (Report, (1,), 'asia')
        assert inspect(newyork_report).identity_token == 'north_america'
        assert inspect(tokyo_report).identity_token == 'asia'
if __name__ == '__main__':
    main()