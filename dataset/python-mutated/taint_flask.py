from flask import Flask, render_template, session, request
from pyappdata.flask import appdata
import settings
import logging
import json
from logging.handlers import RotatingFileHandler
app = Flask(__name__)
app.secret_key = settings.secret_key
app.config.from_object(settings.configClass)
formatter = logging.Formatter(settings.LOG_FORMAT)
handler = RotatingFileHandler(settings.LOG_FILE, maxBytes=settings.LOG_MAX_BYTES, backupCount=settings.LOG_BACKUP_COUNT)
handler.setLevel(logging.getLevelName(settings.LOG_LEVEL))
handler.setFormatter(formatter)
app.logger.addHandler(handler)

def return_error(msg):
    if False:
        while True:
            i = 10
    return render_template('error.htm.j2', msg=msg)

def error(exception=None):
    if False:
        print('Hello World!')
    app.logger.error('Pyappdata error: {}'.format(exception))
    return return_error('Authentication error,\n        please refresh and try again. If this error persists,\n        please contact support.')

@app.route('/launch', methods=['POST', 'GET'])
@appdata(error=error, request='initial', role='any', app=app)
def launch():
    if False:
        print('Hello World!')
    session['person_name_full'] = request.form.get('person_name_full')
    app.logger.info(json.dumps(request.form, indent=2))
    return render_template('launch.htm.j2', person_name_full=session['person_name_full'])

@app.route('/', methods=['GET'])
def index():
    if False:
        while True:
            i = 10
    return render_template('index.htm.j2')
print('Hi')