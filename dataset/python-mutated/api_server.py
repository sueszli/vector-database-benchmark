import logging
from flask import Flask, jsonify, request
from embedchain import App
app = Flask(__name__)

@app.route('/add', methods=['POST'])
def add():
    if False:
        print('Hello World!')
    data = request.get_json()
    data_type = data.get('data_type')
    url_or_text = data.get('url_or_text')
    if data_type and url_or_text:
        try:
            App().add(url_or_text, data_type=data_type)
            return (jsonify({'data': f'Added {data_type}: {url_or_text}'}), 200)
        except Exception:
            logging.exception(f'Failed to add data_type={data_type!r}: url_or_text={url_or_text!r}')
            return (jsonify({'error': f'Failed to add {data_type}: {url_or_text}'}), 500)
    return (jsonify({'error': "Invalid request. Please provide 'data_type' and 'url_or_text' in JSON format."}), 400)

@app.route('/query', methods=['POST'])
def query():
    if False:
        for i in range(10):
            print('nop')
    data = request.get_json()
    question = data.get('question')
    if question:
        try:
            response = App().query(question)
            return (jsonify({'data': response}), 200)
        except Exception:
            logging.exception(f'Failed to query question={question!r}')
            return (jsonify({'error': 'An error occurred. Please try again!'}), 500)
    return (jsonify({'error': "Invalid request. Please provide 'question' in JSON format."}), 400)

@app.route('/chat', methods=['POST'])
def chat():
    if False:
        return 10
    data = request.get_json()
    question = data.get('question')
    if question:
        try:
            response = App().chat(question)
            return (jsonify({'data': response}), 200)
        except Exception:
            logging.exception(f'Failed to chat question={question!r}')
            return (jsonify({'error': 'An error occurred. Please try again!'}), 500)
    return (jsonify({'error': "Invalid request. Please provide 'question' in JSON format."}), 400)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)