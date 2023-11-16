from flask import Flask
app = Flask(__name__)

@app.route('/')
def main():
    if False:
        i = 10
        return i + 15
    raise
app.run(debug=True)
app.run()
app.run(debug=False)
run()
run(debug=True)
run(debug)