import vertexai
from vertexai import language_models

def streaming_prediction(project_id: str, location: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Streaming Chat Example with a Large Language Model.'
    vertexai.init(project=project_id, location=location)
    chat_model = language_models.ChatModel.from_pretrained('chat-bison')
    parameters = {'temperature': 0.8, 'max_output_tokens': 256, 'top_p': 0.95, 'top_k': 40}
    chat = chat_model.start_chat(context='My name is Miles. You are an astronomer, knowledgeable about the solar system.', examples=[language_models.InputOutputTextPair(input_text='How many moons does Mars have?', output_text='The planet Mars has two moons, Phobos and Deimos.')])
    responses = chat.send_message_streaming(message='How many planets are there in the solar system?', **parameters)
    results = []
    for response in responses:
        print(response)
        results.append(str(response))
    results = ''.join(results)
    return results
if __name__ == '__main__':
    streaming_prediction()