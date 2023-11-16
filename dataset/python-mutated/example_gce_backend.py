import platform
import flask
import validate_jwt
CLOUD_PROJECT_ID = 'YOUR_PROJECT_ID'
BACKEND_SERVICE_ID = 'YOUR_BACKEND_SERVICE_ID'
app = flask.Flask(__name__)

@app.route('/')
def root():
    if False:
        return 10
    jwt = flask.request.headers.get('x-goog-iap-jwt-assertion')
    if jwt is None:
        return 'Unauthorized request.'
    expected_audience = f'/projects/{CLOUD_PROJECT_ID}/global/backendServices/{BACKEND_SERVICE_ID}'
    (user_id, user_email, error_str) = validate_jwt.validate_iap_jwt(jwt, expected_audience)
    if error_str:
        return f'Error: {error_str}'
    else:
        return f'Hi, {user_email}. I am {platform.node()}.'
if __name__ == '__main__':
    app.run()