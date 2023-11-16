from flask import Flask, request
app = Flask(__name__)

@app.route('/return/request/url', methods=['GET', 'POST'])
def return_request_url():
    if False:
        for i in range(10):
            print('nop')
    return request.url