from sqlalchemy.exc import SQLAlchemyError
from lwe.backends.api.orm import Manager

class ConversationManager(Manager):

    def get_conversations(self, user_id, limit=None, offset=None, order_desc=True):
        if False:
            while True:
                i = 10
        try:
            user = self.orm_get_user(user_id)
            conversations = self.orm_get_conversations(user, limit, offset, order_desc)
            return (True, conversations, 'Conversations retrieved successfully.')
        except SQLAlchemyError as e:
            return self._handle_error(f'Failed to retrieve conversations: {str(e)}')

    def add_conversation(self, user_id, title=None, hidden=False):
        if False:
            for i in range(10):
                print('nop')
        try:
            user = self.orm_get_user(user_id)
            conversation = self.orm_add_conversation(user, title, hidden)
            return (True, conversation, 'Conversation created successfully.')
        except SQLAlchemyError as e:
            return self._handle_error(f'Failed to create conversation: {str(e)}')

    def get_conversation(self, conversation_id):
        if False:
            print('Hello World!')
        try:
            conversation = self.orm_get_conversation(conversation_id)
            if conversation:
                return (True, conversation, 'Conversation retrieved successfully.')
            else:
                return (False, None, 'Conversation not found.')
        except SQLAlchemyError as e:
            return self._handle_error(f'Failed to retrieve conversation: {str(e)}')

    def edit_conversation(self, conversation_id, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (success, conversation, message) = self.get_conversation(conversation_id)
        if not success:
            return (success, conversation, message)
        if not conversation:
            return (False, None, 'Conversation not found')
        try:
            updated_conversation = self.orm_edit_conversation(conversation, **kwargs)
        except SQLAlchemyError as e:
            return self._handle_error(f'Failed to edit conversation: {str(e)}')
        return (True, updated_conversation, 'Conversation edited successfully')

    def edit_conversation_title(self, conversation_id, new_title):
        if False:
            i = 10
            return i + 15
        (success, conversation, message) = self.get_conversation(conversation_id)
        if not success:
            return (success, conversation, message)
        try:
            updated_conversation = self.orm_edit_conversation(conversation, title=new_title)
        except SQLAlchemyError as e:
            return self._handle_error(f'Failed to update conversation title: {str(e)}')
        return (True, updated_conversation, 'Conversation title updated successfully.')

    def hide_conversation(self, conversation_id):
        if False:
            return 10
        (success, conversation, message) = self.get_conversation(conversation_id)
        if not success:
            return (success, conversation, message)
        try:
            updated_conversation = self.orm_edit_conversation(conversation, hidden=True)
        except SQLAlchemyError as e:
            return self._handle_error(f'Failed to hide conversation: {str(e)}')
        return (True, updated_conversation, 'Conversation hidden successfully.')

    def unhide_conversation(self, conversation_id):
        if False:
            return 10
        (success, conversation, message) = self.get_conversation(conversation_id)
        if not success:
            return (success, conversation, message)
        try:
            updated_conversation = self.orm_edit_conversation(conversation, hidden=False)
        except SQLAlchemyError as e:
            return self._handle_error(f'Failed to unhide conversation: {str(e)}')
        return (True, updated_conversation, 'Conversation unhidden successfully.')

    def delete_conversation(self, conversation_id):
        if False:
            print('Hello World!')
        (success, conversation, message) = self.get_conversation(conversation_id)
        if not success:
            return (success, conversation, message)
        try:
            self.orm_delete_conversation(conversation)
        except SQLAlchemyError as e:
            return self._handle_error(f'Failed to delete conversation: {str(e)}')
        return (True, None, 'Conversation deleted successfully.')