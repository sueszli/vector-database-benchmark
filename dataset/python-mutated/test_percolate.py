from .percolate import BlogPost, setup

def test_post_gets_tagged_automatically(write_client):
    if False:
        while True:
            i = 10
    setup()
    bp = BlogPost(_id=47, content='nothing about snakes here!')
    bp_py = BlogPost(_id=42, content='something about Python here!')
    bp.save()
    bp_py.save()
    assert [] == bp.tags
    assert {'programming', 'development', 'python'} == set(bp_py.tags)