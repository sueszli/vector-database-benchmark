from coalib.bearlib.aspects import Root
from coalib.bearlib.aspects.docs import Documentation

class aspectsDocsTest:

    def test_aspects_docs(self):
        if False:
            while True:
                i = 10

        def check(aspects):
            if False:
                while True:
                    i = 10
            for aspect in aspects:
                assert isinstance(aspect.docs, Documentation)
                assert aspect.docs.check_consistency()
                check(aspect.subaspects.values())
        check(Root.subaspects.values())