import vertexai
from vertexai import language_models

def streaming_prediction(project_id: str, location: str) -> str:
    if False:
        print('Hello World!')
    'Streaming Code Chat Example with a Large Language Model.'
    vertexai.init(project=project_id, location=location)
    codechat_model = language_models.CodeChatModel.from_pretrained('codechat-bison')
    parameters = {'temperature': 0.8, 'max_output_tokens': 1024}
    codechat = codechat_model.start_chat()
    responses = codechat.send_message_streaming(message='Please help write a function to calculate the min of two numbers', **parameters)
    results = []
    for response in responses:
        print(response)
        results.append(str(response))
    results = '\n'.join(results)
    return results
if __name__ == '__main__':
    streaming_prediction()