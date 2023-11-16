import os
import time
from googleapiclient.discovery import build
client_service = build('jobs', 'v3')
parent = 'projects/' + os.environ['GOOGLE_CLOUD_PROJECT']

def search_for_alerts(client_service, company_name):
    if False:
        print('Hello World!')
    request_metadata = {'user_id': 'HashedUserId', 'session_id': 'HashedSessionId', 'domain': 'www.google.com'}
    request = {'search_mode': 'JOB_SEARCH', 'request_metadata': request_metadata}
    if company_name is not None:
        request.update({'job_query': {'company_names': [company_name]}})
    response = client_service.projects().jobs().searchForAlert(parent=parent, body=request).execute()
    print(response)

def set_up():
    if False:
        i = 10
        return i + 15
    import base_company_sample
    import base_job_sample
    company_to_be_created = base_company_sample.generate_company()
    company_created = base_company_sample.create_company(client_service, company_to_be_created)
    company_name = company_created.get('name')
    job_to_be_created = base_job_sample.generate_job_with_required_fields(company_name)
    job_name = base_job_sample.create_job(client_service, job_to_be_created).get('name')
    return (company_name, job_name)

def tear_down(company_name, job_name):
    if False:
        return 10
    import base_company_sample
    import base_job_sample
    base_job_sample.delete_job(client_service, job_name)
    base_company_sample.delete_company(client_service, company_name)

def run_sample(company_name):
    if False:
        while True:
            i = 10
    search_for_alerts(client_service, company_name)
if __name__ == '__main__':
    (company_name, job_name) = set_up()
    time.sleep(10)
    run_sample(company_name)
    tear_down(company_name, job_name)