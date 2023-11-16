from chatterbot import ChatBot
from chatterbot.conversation import Statement
'\nThis example shows how to create a chat bot that\nwill learn responses based on an additional feedback\nelement from the user.\n'
bot = ChatBot('Feedback Learning Bot', storage_adapter='chatterbot.storage.SQLStorageAdapter')

def get_feedback():
    if False:
        print('Hello World!')
    text = input()
    if 'yes' in text.lower():
        return True
    elif 'no' in text.lower():
        return False
    else:
        print('Please type either "Yes" or "No"')
        return get_feedback()
print('Type something to begin...')
while True:
    try:
        input_statement = Statement(text=input())
        response = bot.generate_response(input_statement)
        print('\n Is "{}" a coherent response to "{}"? \n'.format(response.text, input_statement.text))
        if get_feedback() is False:
            print('please input the correct one')
            correct_response = Statement(text=input())
            bot.learn_response(correct_response, input_statement)
            print('Responses added to bot!')
    except (KeyboardInterrupt, EOFError, SystemExit):
        break