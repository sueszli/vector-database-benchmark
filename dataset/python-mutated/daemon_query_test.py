import testslide
from ..daemon_query import InvalidQueryResponse, Response

class ResponseTest(testslide.TestCase):

    def test_parse_response(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def assert_parsed(text: str, expected: Response) -> None:
            if False:
                print('Hello World!')
            self.assertEqual(Response.parse(text), expected)

        def assert_not_parsed(text: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            with self.assertRaises(InvalidQueryResponse):
                Response.parse(text)
        assert_not_parsed('42')
        assert_not_parsed('derp')
        assert_not_parsed('{}')
        assert_not_parsed('[]')
        assert_not_parsed('["Query"]')
        assert_parsed('["Query", []]', Response(payload=[]))
        assert_parsed('["Query",{"response":{"boolean":true}}]', Response(payload={'response': {'boolean': True}}))
        assert_parsed('["Query", {"response":[{"object":[]}]}]', Response(payload={'response': [{'object': []}]}))
        assert_parsed('["Query",{"response":{"path":"/foo/bar.py"}}]', Response(payload={'response': {'path': '/foo/bar.py'}}))