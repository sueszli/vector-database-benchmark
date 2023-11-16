import strawberry

def test_can_instantiate_types_directly():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class User:
        username: str

        @strawberry.field
        def email(self) -> str:
            if False:
                print('Hello World!')
            return self.username + '@somesite.com'
    user = User(username='abc')
    assert user.username == 'abc'
    assert user.email() == 'abc@somesite.com'