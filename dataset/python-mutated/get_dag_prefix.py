"""Get a Cloud Composer environment via the REST API.

This code sample gets a Cloud Composer environment resource and prints the
Cloud Storage path used to store Apache Airflow DAGs.
"""
import argparse

def get_dag_prefix(project_id, location, composer_environment):
    if False:
        while True:
            i = 10
    import google.auth
    import google.auth.transport.requests
    (credentials, _) = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    authed_session = google.auth.transport.requests.AuthorizedSession(credentials)
    environment_url = 'https://composer.googleapis.com/v1beta1/projects/{}/locations/{}/environments/{}'.format(project_id, location, composer_environment)
    response = authed_session.request('GET', environment_url)
    environment_data = response.json()
    print(environment_data['config']['dagGcsPrefix'])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Project ID.')
    parser.add_argument('location', help='Region of the Cloud Composer environment.')
    parser.add_argument('composer_environment', help='Name of the Cloud Composer environment.')
    args = parser.parse_args()
    get_dag_prefix(args.project_id, args.location, args.composer_environment)