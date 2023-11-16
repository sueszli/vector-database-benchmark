"""Get the client ID associated with a Cloud Composer environment."""
import argparse

def get_client_id(project_id, location, composer_environment):
    if False:
        i = 10
        return i + 15
    import google.auth
    import google.auth.transport.requests
    import requests
    import six.moves.urllib.parse
    (credentials, _) = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    authed_session = google.auth.transport.requests.AuthorizedSession(credentials)
    environment_url = 'https://composer.googleapis.com/v1beta1/projects/{}/locations/{}/environments/{}'.format(project_id, location, composer_environment)
    composer_response = authed_session.request('GET', environment_url)
    environment_data = composer_response.json()
    composer_version = environment_data['config']['softwareConfig']['imageVersion']
    if 'composer-1' not in composer_version:
        version_error = 'This script is intended to be used with Composer 1 environments. In Composer 2, the Airflow Webserver is not in the tenant project, so there is no tenant client ID. See https://cloud.google.com/composer/docs/composer-2/environment-architecture for more details.'
        raise RuntimeError(version_error)
    airflow_uri = environment_data['config']['airflowUri']
    redirect_response = requests.get(airflow_uri, allow_redirects=False)
    redirect_location = redirect_response.headers['location']
    parsed = six.moves.urllib.parse.urlparse(redirect_location)
    query_string = six.moves.urllib.parse.parse_qs(parsed.query)
    print(query_string['client_id'][0])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Project ID.')
    parser.add_argument('location', help='Region of the Cloud Composer environment.')
    parser.add_argument('composer_environment', help='Name of the Cloud Composer environment.')
    args = parser.parse_args()
    get_client_id(args.project_id, args.location, args.composer_environment)