from .constants import ConversationRole

class ConversationTurn(object):

    def __init__(self, role: ConversationRole, name=None, message='', full_response=None, request=None):
        if False:
            print('Hello World!')
        self.role = role
        self.name = name
        self.message = message
        self.full_response = full_response
        self.request = request

    def to_openai_chat_format(self, reverse=False):
        if False:
            while True:
                i = 10
        if reverse is False:
            return {'role': self.role.value, 'content': self.message}
        elif self.role == ConversationRole.ASSISTANT:
            return {'role': ConversationRole.USER.value, 'content': self.message}
        else:
            return {'role': ConversationRole.ASSISTANT.value, 'content': self.message}

    def to_annotation_format(self, turn_number: int):
        if False:
            for i in range(10):
                print('nop')
        return {'turn_number': turn_number, 'response': self.message, 'actor': self.role.value if self.name is None else self.name, 'request': self.request, 'full_json_response': self.full_response}

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return f'({self.role.value}): {self.message}'

    def __repr__(self) -> str:
        if False:
            return 10
        return f'CoversationTurn(role={self.role.value}, message={self.message})'