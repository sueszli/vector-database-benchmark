import json

def convert_to_openai_messages(messages, function_calling=True):
    if False:
        return 10
    new_messages = []
    for message in messages:
        new_message = {'role': message['role'], 'content': ''}
        if 'message' in message and 'image' not in message:
            new_message['content'] = message['message']
        if 'code' in message:
            if function_calling:
                new_message['function_call'] = {'name': 'execute', 'arguments': json.dumps({'language': message['language'], 'code': message['code']}), 'parsed_arguments': {'language': message['language'], 'code': message['code']}}
            else:
                new_message['content'] += f"\n\n```{message['language']}\n{message['code']}\n```"
                new_message['content'] = new_message['content'].strip()
        new_messages.append(new_message)
        if 'output' in message:
            if function_calling:
                new_messages.append({'role': 'function', 'name': 'execute', 'content': message['output']})
            else:
                if message['output'] == 'No output':
                    content = 'The code above was executed on my machine. It produced no output. Was that expected?'
                else:
                    content = 'Code output: ' + message['output'] + "\n\nWhat does this output mean / what's next (if anything)?"
                new_messages.append({'role': 'user', 'content': content})
        if 'image' in message:
            new_message = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': message['image'], 'detail': 'high'}}]}
            if message['role'] == 'user':
                if 'message' in message:
                    new_message['content'].append({'type': 'text', 'text': message['message']})
                    new_message['content'].reverse()
                new_messages.append(new_message)
            if message['role'] == 'assistant':
                if message == messages[-1]:
                    new_message['content'].append({'type': 'text', 'text': "This is the result. Does that look right? Could it be closer to what we're aiming for, or is it done? Be detailed in exactly how we could improve it first, then write code to improve it. Unless you think it's done (I might agree)!"})
                    new_message['content'].reverse()
                new_messages.append(new_message)
                if 'output' in message and message == messages[-1]:
                    pass
    if not function_calling:
        new_messages = [msg for msg in new_messages if 'content' in msg and len(msg['content']) != 0]
    return new_messages