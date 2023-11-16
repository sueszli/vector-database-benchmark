from . import BaseTest

class TestSQP(BaseTest):

    def test_search_query_parser(self):
        if False:
            while True:
                i = 10
        from kitty.search_query_parser import ParseException, search
        locations = 'id'
        universal_set = {1, 2, 3, 4, 5}

        def get_matches(location, query, candidates):
            if False:
                i = 10
                return i + 15
            return {x for x in candidates if query == str(x)}

        def t(q, expected=set()):
            if False:
                return 10
            actual = search(q, locations, universal_set, get_matches)
            self.ae(actual, expected)
        t('id:1', {1})
        t('id:"1"', {1})
        t('id:1 and id:1', {1})
        t('id:1 or id:2', {1, 2})
        t('id:1 and id:2')
        t('not id:1', universal_set - {1})
        t('(id:1 or id:2) and id:1', {1})
        self.assertRaises(ParseException, t, '1')
        self.assertRaises(ParseException, t, '"id:1"')