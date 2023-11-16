from . import alias_migration
from .alias_migration import ALIAS, PATTERN, BlogPost, migrate

def test_alias_migration(write_client):
    if False:
        while True:
            i = 10
    alias_migration.setup()
    assert write_client.indices.exists_template(name=ALIAS)
    assert write_client.indices.exists(index=PATTERN)
    assert write_client.indices.exists_alias(name=ALIAS)
    indices = write_client.indices.get(index=PATTERN)
    assert len(indices) == 1
    (index_name, _) = indices.popitem()
    with open(__file__) as f:
        bp = BlogPost(_id=0, title='Hello World!', tags=['testing', 'dummy'], content=f.read())
        bp.save(refresh=True)
    assert BlogPost.search().count() == 1
    bp = BlogPost.search().execute()[0]
    assert isinstance(bp, BlogPost)
    assert not bp.is_published()
    assert '0' == bp.meta.id
    migrate()
    indices = write_client.indices.get(index=PATTERN)
    assert 2 == len(indices)
    alias = write_client.indices.get(index=ALIAS)
    assert 1 == len(alias)
    assert index_name not in alias
    assert BlogPost.search().count() == 1
    bp = BlogPost.search().execute()[0]
    assert isinstance(bp, BlogPost)
    assert '0' == bp.meta.id