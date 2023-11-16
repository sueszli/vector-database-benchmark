from vertexai.language_models import CodeChatModel

def write_a_function(temperature: float=0.5) -> object:
    if False:
        print('Hello World!')
    'Example of using Codey for Code Chat Model to write a function.'
    parameters = {'temperature': temperature, 'max_output_tokens': 1024}
    code_chat_model = CodeChatModel.from_pretrained('codechat-bison@001')
    chat = code_chat_model.start_chat()
    response = chat.send_message('Please help write a function to calculate the min of two numbers', **parameters)
    print(f'Response from Model: {response.text}')
    return response
if __name__ == '__main__':
    write_a_function()