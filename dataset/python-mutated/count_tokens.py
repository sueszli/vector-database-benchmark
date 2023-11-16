import tiktoken
from litellm import cost_per_token

def count_tokens(text='', model='gpt-4'):
    if False:
        return 10
    '\n    Count the number of tokens in a string\n    '
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

def token_cost(tokens=0, model='gpt-4'):
    if False:
        print('Hello World!')
    '\n    Calculate the cost of the current number of tokens\n    '
    (prompt_cost, _) = cost_per_token(model=model, prompt_tokens=tokens)
    return round(prompt_cost, 6)

def count_messages_tokens(messages=[], model=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Count the number of tokens in a list of messages\n    '
    tokens_used = 0
    for message in messages:
        if isinstance(message, str):
            tokens_used += count_tokens(message, model=model)
        elif 'message' in message:
            tokens_used += count_tokens(message['message'], model=model)
            if 'code' in message:
                tokens_used += count_tokens(message['code'], model=model)
            if 'output' in message:
                tokens_used += count_tokens(message['output'], model=model)
    prompt_cost = token_cost(tokens_used, model=model)
    return (tokens_used, prompt_cost)