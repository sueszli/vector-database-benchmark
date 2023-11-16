def test_block_fallback():
    if False:
        print('Hello World!')

    class ByContains(object):

        def __contains__(self, other):
            if False:
                return 10
            return False
    c = ByContains()

    class BlockContains(ByContains):
        """Is not a container

        This class is a perfectly good iterable (as tested by
        list(bc)), as well as inheriting from a perfectly good
        container, but __contains__ = None prevents the usual
        fallback to iteration in the container protocol. That
        is, normally, 0 in bc would fall back to the equivalent
        of any(x==0 for x in bc), but here it's blocked from
        doing so.
        """

        def __iter__(self):
            if False:
                return 10
            while False:
                yield None
        __contains__ = None
    bc = BlockContains()
    assert not 0 in c
    assert not 0 in list(bc)
test_block_fallback()