from celery.utils.objects import Bunch

class test_Bunch:

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        x = Bunch(foo='foo', bar=2)
        assert x.foo == 'foo'
        assert x.bar == 2