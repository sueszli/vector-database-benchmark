import openai
openai.api_key = ''
openai.api_base = 'http://localhost:1337'

def main():
    if False:
        print('Hello World!')
    chat_completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role': 'user', 'content': 'write a poem about a tree'}], stream=True)
    if isinstance(chat_completion, dict):
        print(chat_completion.choices[0].message.content)
    else:
        for token in chat_completion:
            content = token['choices'][0]['delta'].get('content')
            if content != None:
                print(content, end='', flush=True)
if __name__ == '__main__':
    main()