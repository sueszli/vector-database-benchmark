from pytest import mark
from graphql_relay.utils import base64
from graphene.types import ObjectType, Schema, String
from graphene.relay.connection import Connection, ConnectionField, PageInfo
from graphene.relay.node import Node
letter_chars = ['A', 'B', 'C', 'D', 'E']

class Letter(ObjectType):

    class Meta:
        interfaces = (Node,)
    letter = String()

class LetterConnection(Connection):

    class Meta:
        node = Letter

class Query(ObjectType):
    letters = ConnectionField(LetterConnection)
    connection_letters = ConnectionField(LetterConnection)
    async_letters = ConnectionField(LetterConnection)
    node = Node.Field()

    def resolve_letters(self, info, **args):
        if False:
            print('Hello World!')
        return list(letters.values())

    async def resolve_async_letters(self, info, **args):
        return list(letters.values())

    def resolve_connection_letters(self, info, **args):
        if False:
            i = 10
            return i + 15
        return LetterConnection(page_info=PageInfo(has_next_page=True, has_previous_page=False), edges=[LetterConnection.Edge(node=Letter(id=0, letter='A'), cursor='a-cursor')])
schema = Schema(Query)
letters = {letter: Letter(id=i, letter=letter) for (i, letter) in enumerate(letter_chars)}

def edges(selected_letters):
    if False:
        print('Hello World!')
    return [{'node': {'id': base64('Letter:%s' % letter.id), 'letter': letter.letter}, 'cursor': base64('arrayconnection:%s' % letter.id)} for letter in [letters[i] for i in selected_letters]]

def cursor_for(ltr):
    if False:
        for i in range(10):
            print('nop')
    letter = letters[ltr]
    return base64('arrayconnection:%s' % letter.id)

def execute(args=''):
    if False:
        print('Hello World!')
    if args:
        args = '(' + args + ')'
    return schema.execute('\n    {\n        letters%s {\n            edges {\n                node {\n                    id\n                    letter\n                }\n                cursor\n            }\n            pageInfo {\n                hasPreviousPage\n                hasNextPage\n                startCursor\n                endCursor\n            }\n        }\n    }\n    ' % args)

@mark.asyncio
async def test_connection_async():
    result = await schema.execute_async('\n    {\n        asyncLetters(first:1) {\n            edges {\n                node {\n                    id\n                    letter\n                }\n            }\n            pageInfo {\n                hasPreviousPage\n                hasNextPage\n            }\n        }\n    }\n    ')
    assert not result.errors
    assert result.data == {'asyncLetters': {'edges': [{'node': {'id': 'TGV0dGVyOjA=', 'letter': 'A'}}], 'pageInfo': {'hasPreviousPage': False, 'hasNextPage': True}}}