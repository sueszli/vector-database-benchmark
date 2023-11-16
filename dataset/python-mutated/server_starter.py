from __future__ import print_function, division
from builtins import range
import numpy as np
from flask import Flask, jsonify, request
from scipy.stats import beta
app = Flask(__name__)

class Bandit:

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.name = name

    def sample(self):
        if False:
            while True:
                i = 10
        return 1
banditA = Bandit('A')
banditB = Bandit('B')

@app.route('/get_ad')
def get_ad():
    if False:
        while True:
            i = 10
    return jsonify({'advertisement_id': 'A'})

@app.route('/click_ad', methods=['POST'])
def click_ad():
    if False:
        i = 10
        return i + 15
    result = 'OK'
    if request.form['advertisement_id'] == 'A':
        pass
    elif request.form['advertisement_id'] == 'B':
        pass
    else:
        result = 'Invalid Input.'
    return jsonify({'result': result})
if __name__ == '__main__':
    app.run(host='127.0.0.1', port='8888')