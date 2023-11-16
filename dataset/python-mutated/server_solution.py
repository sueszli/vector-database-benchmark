from __future__ import print_function, division
from builtins import range
import numpy as np
from flask import Flask, jsonify, request
from scipy.stats import beta
app = Flask(__name__)

class Bandit:

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.clks = 0
        self.views = 0
        self.name = name

    def sample(self):
        if False:
            while True:
                i = 10
        a = 1 + self.clks
        b = 1 + self.views - self.clks
        return np.random.beta(a, b)

    def add_click(self):
        if False:
            return 10
        self.clks += 1

    def add_view(self):
        if False:
            return 10
        self.views += 1
        if self.views % 50 == 0:
            print('%s: clks=%s, views=%s' % (self.name, self.clks, self.views))
banditA = Bandit('A')
banditB = Bandit('B')

@app.route('/get_ad')
def get_ad():
    if False:
        return 10
    if banditA.sample() > banditB.sample():
        ad = 'A'
        banditA.add_view()
    else:
        ad = 'B'
        banditB.add_view()
    return jsonify({'advertisement_id': ad})

@app.route('/click_ad', methods=['POST'])
def click_ad():
    if False:
        print('Hello World!')
    result = 'OK'
    if request.form['advertisement_id'] == 'A':
        banditA.add_click()
    elif request.form['advertisement_id'] == 'B':
        banditB.add_click()
    else:
        result = 'Invalid Input.'
    return jsonify({'result': result})
if __name__ == '__main__':
    app.run(host='127.0.0.1', port='8888')