import sqlalchemy as sa
from buildbot.util import sautils

def test_unicode(migrate_engine):
    if False:
        while True:
            i = 10
    'Test that the database can handle inserting and selecting Unicode'
    submeta = sa.MetaData()
    submeta.bind = migrate_engine
    test_unicode = sautils.Table('test_unicode', submeta, sa.Column('u', sa.Unicode(length=100)), sa.Column('b', sa.LargeBinary))
    test_unicode.create()
    u = 'Frosty the ☃'
    b = b'\xff\xff\x00'
    ins = test_unicode.insert().values(u=u, b=b)
    migrate_engine.execute(ins)
    row = migrate_engine.execute(sa.select([test_unicode])).fetchall()[0]
    assert isinstance(row['u'], str)
    assert row['u'] == u
    assert isinstance(row['b'], bytes)
    assert row['b'] == b
    test_unicode.drop()